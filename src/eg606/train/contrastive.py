"""Train the contrastive EEG-audio matcher on speech or music.

    python -m eg606.train.contrastive --dataset speech --epochs 40
    python -m eg606.train.contrastive --dataset music --init <ckpt>     # speech -> music transfer

Batches are drawn from a handful of recordings at a time, so the in-batch negatives are mostly
other moments of the same recording: slow drift cannot separate them, only real time-locking can.
Evaluation is the same match-mismatch task used by the linear baseline, on held-out listeners.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from eg606.models.contrastive import Matcher, match_mismatch_accuracy
from eg606.paths import DATA_ROOT, derived_dir

log = logging.getLogger(__name__)
FS = 64
SUB = 5 * FS          # the window the model is trained on
GUARD = 1 * FS        # imposter offset, matching the linear protocol


def zscore(x, axis=-1):
    m, s = x.mean(axis=axis, keepdims=True), x.std(axis=axis, keepdims=True)
    return (x - m) / np.where(s > 0, s, 1.0)


def load_speech(feature: str, limit: int | None) -> list[dict]:
    root = derived_dir("sparrkulee")
    feats = dict(np.load(root / "audio_features.npz"))
    out = []
    for p in sorted(root.glob("sub-*.npz"))[:limit]:
        d = np.load(p, allow_pickle=True)
        for m in json.loads(str(d["meta"])):
            y = feats.get(f"{feature}::{m['stimulus']}")
            if y is None:
                continue
            eeg = d[m["key"]].astype(np.float32)
            n = min(eeg.shape[1], len(y))
            out.append({"sub": p.stem, "eeg": zscore(eeg[:, :n]),
                        "y": zscore(y[:n].astype(np.float32))[None]})
    return out


def apply_shift(eeg: np.ndarray, y: np.ndarray, shift: int):
    """Assume the response is `shift` samples later than the marker says: pair EEG(t+shift) with y(t).

    MUSIN-G needs +62 ms (4 samples at 64 Hz); Bach and SparrKULee need none (see LAB_NOTEBOOK).
    """
    if shift <= 0:
        return eeg, y
    n = min(eeg.shape[1] - shift, y.shape[-1] - shift)
    return eeg[:, shift: shift + n], y[..., :n]


def load_bach(feature: str = "onset", shift: int = 0) -> list[dict]:
    root = derived_dir("bach_silence")
    out = []
    for p in sorted(root.glob("sub-*.npz")):
        d = np.load(p, allow_pickle=True)
        for m in json.loads(str(d["meta"])):
            eeg = d[f"eeg{m['trial']:02d}"].astype(np.float32)
            y = d[f"{feature}{m['trial']:02d}"].astype(np.float32)[None]
            n = min(eeg.shape[1], y.shape[-1])
            eeg, y = apply_shift(eeg[:, :n], y[..., :n], shift)
            out.append({"sub": p.stem, "song": m["chorale"], "eeg": zscore(eeg), "y": zscore(y)})
    return out


def load_music(feature: str, variant: str = "bs64", shift: int = 0) -> list[dict]:
    root = derived_dir("musin_g")
    feats = dict(np.load(root / "audio_features.npz"))
    pattern = re.compile(rf"^sub-\d+_{re.escape(variant)}$")
    out = []
    for p in sorted(x for x in root.glob("sub-*.npz") if pattern.match(x.stem)):
        d = np.load(p, allow_pickle=True)
        for m in json.loads(str(d["meta"])):
            song = m["song"]
            eeg = d[f"song{song:02d}"].astype(np.float32)
            y = feats[f"{feature}{song:02d}"].astype(np.float32)
            n = min(eeg.shape[1], len(y))
            eeg, y = apply_shift(eeg[:, :n], y[:n][None], shift)
            out.append({"sub": p.stem.split("_")[0], "song": song,
                        "eeg": zscore(eeg), "y": zscore(y)})
    return out


def sample_batch(recs, rng, batch: int, per_rec: int, device):
    """Draw windows from a few recordings so in-batch negatives are mostly same-recording."""
    eeg_b, aud_b = [], []
    for _ in range(max(1, batch // per_rec)):
        r = recs[rng.integers(len(recs))]
        n = r["eeg"].shape[1]
        if n <= SUB + 1:
            continue
        for s in rng.integers(0, n - SUB, size=per_rec):
            eeg_b.append(r["eeg"][:, s:s + SUB])
            aud_b.append(r["y"][:, s:s + SUB])
    eeg = torch.from_numpy(np.stack(eeg_b)).to(device, non_blocking=True)
    aud = torch.from_numpy(np.stack(aud_b)).to(device, non_blocking=True)
    return eeg, aud


@torch.no_grad()
def evaluate(model, recs, windows, device, max_trials: int | None = None) -> dict:
    out = {}
    trials = recs[:max_trials] if max_trials else recs
    for w in windows:
        hits = total = 0
        for r in trials:
            h, t = match_mismatch_accuracy(model, torch.from_numpy(r["eeg"]),
                                           torch.from_numpy(r["y"]), int(w * FS), SUB, GUARD, device)
            hits += h
            total += t
        out[f"{w:g}s"] = hits / total if total else float("nan")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.train.contrastive")
    ap.add_argument("--dataset", default="speech", choices=["speech", "music"])
    ap.add_argument("--feature", default="onset", choices=["onset", "env"])
    ap.add_argument("--holdout", type=int, default=4, help="listeners kept for evaluation")
    ap.add_argument("--subjects", type=int, default=None, help="limit listeners (speech)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=400, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--per-rec", type=int, default=32, help="windows drawn per recording")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--init", default=None, help="checkpoint to start from (transfer)")
    ap.add_argument("--eval-music", action="store_true", help="also score music each epoch")
    ap.add_argument("--windows", default="5,10,30")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows = [float(w) for w in args.windows.split(",")]

    t0 = time.time()
    recs = load_speech(args.feature, args.subjects) if args.dataset == "speech" else load_music(args.feature)
    subs = sorted({r["sub"] for r in recs})
    held = set(subs[-args.holdout:]) if args.holdout else set()
    train = [r for r in recs if r["sub"] not in held]
    test = [r for r in recs if r["sub"] in held]
    log.info("%s: %d recordings (%d listeners), train %d / test %d, loaded in %.0fs",
             args.dataset, len(recs), len(subs), len(train), len(test), time.time() - t0)

    music_eval = None
    if args.eval_music and args.dataset == "speech":
        music_eval = load_music(args.feature)
        log.info("music evaluation set: %d trials", len(music_eval))

    n_ch = train[0]["eeg"].shape[0]
    model = Matcher(n_channels=n_ch, dim=args.dim).to(device)
    if args.init:
        state = torch.load(args.init, map_location=device)
        model.load_state_dict(state["model"])
        log.info("initialised from %s (epoch %d)", args.init, state.get("epoch", -1))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * args.steps)

    ckpt_dir = DATA_ROOT / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_path = Path("results") / f"train_{args.tag}.json"
    res_path.parent.mkdir(exist_ok=True)
    history, best = [], -np.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _ in range(args.steps):
            eeg, aud = sample_batch(train, rng, args.batch, args.per_rec, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model.loss(eeg, aud)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())

        entry = {"epoch": epoch, "loss": float(np.mean(losses))}
        entry["holdout"] = evaluate(model, test, windows, device, max_trials=40)
        if music_eval is not None:
            entry["music_zero_shot"] = evaluate(model, music_eval, windows, device, max_trials=60)
        history.append(entry)
        log.info("epoch %2d  loss %.4f  holdout %s%s", epoch, entry["loss"],
                 {k: round(v, 3) for k, v in entry["holdout"].items()},
                 "  music " + str({k: round(v, 3) for k, v in entry["music_zero_shot"].items()})
                 if music_eval is not None else "")

        score = entry["holdout"].get(f"{windows[-1]:g}s", float("nan"))
        torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                   ckpt_dir / f"{args.tag}_last.pt")
        if np.isfinite(score) and score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                       ckpt_dir / f"{args.tag}_best.pt")
        res_path.write_text(json.dumps({"args": vars(args), "history": history}, indent=1))

    print(f"\n=== {args.tag}: best holdout {windows[-1]:g}s = {best:.3f} (chance 0.5)")
    print(f"checkpoints: {ckpt_dir}/{args.tag}_best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
