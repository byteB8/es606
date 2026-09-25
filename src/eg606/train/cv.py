"""Honest evaluation of the deep model: listener K-fold cross-validation.

For every fold the test listeners are never seen, and the epoch is chosen on two *separate*
validation listeners — never on the test set (v1 chose its best epoch on the test listeners, which
flattered it). Results are per listener, so the statistics match the linear baseline.

    python -m eg606.train.cv --dataset music --model v2 --cv 5
    python -m eg606.train.cv --dataset speech --model v2 --tag speech_v2      # pretraining
    python -m eg606.train.cv --dataset music --model v2 --cv 5 --init <ckpt>  # transfer
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from eg606.models.contrastive import Matcher, TimeMatcher, match_mismatch_accuracy
from eg606.paths import DATA_ROOT
from eg606.train.contrastive import GUARD, SUB, FS, load_bach, load_music, load_speech, sample_batch

log = logging.getLogger(__name__)


def build(kind: str, n_ch: int, dim: int, channel_drop: float):
    return TimeMatcher(n_channels=n_ch, dim=dim, channel_drop=channel_drop) if kind == "v2" \
        else Matcher(n_channels=n_ch, dim=dim)


@torch.no_grad()
def per_listener(model, recs, windows, device):
    out = {}
    for sub in sorted({r["sub"] for r in recs}):
        row = {}
        for w in windows:
            hits = total = 0
            for r in (x for x in recs if x["sub"] == sub):
                h, t = match_mismatch_accuracy(model, torch.from_numpy(r["eeg"]),
                                               torch.from_numpy(r["y"]), int(w * FS), SUB, GUARD, device)
                hits += h
                total += t
            row[f"{w:g}s"] = hits / total if total else float("nan")
        out[sub] = row
    return out


def train_one(train, val, args, device, rng, init=None):
    model = build(args.model, train[0]["eeg"].shape[0], args.dim, args.channel_drop).to(device)
    if init:
        model.load_state_dict(torch.load(init, map_location=device)["model"], strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * args.steps)
    windows = [float(w) for w in args.windows.split(",")]
    best, best_state, best_epoch, stale = -np.inf, None, 0, 0
    history = []
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
        v = per_listener(model, val, windows, device)
        vscore = float(np.nanmean([np.nanmean(list(r.values())) for r in v.values()]))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val": vscore})
        log.info("  epoch %2d loss %.4f val %.3f", epoch, np.mean(losses), vscore)
        if vscore > best:
            best, best_state, best_epoch, stale = vscore, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= args.patience:
                log.info("  early stop at epoch %d (best %d)", epoch, best_epoch)
                break
    model.load_state_dict(best_state)
    return model, best_epoch, history


def summarise(results, windows):
    lines = []
    for w in windows:
        k = f"{w:g}s"
        a = np.array([r[k] for r in results.values()])
        t, p = stats.ttest_1samp(a, 0.5)
        lines.append(f"  {k:>4}: {a.mean():.3f} +/- {a.std():.3f}  ({(a > 0.5).sum()}/{len(a)} above chance)"
                     f"  t={t:.2f} p={p:.2g}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.train.cv")
    ap.add_argument("--dataset", default="music", choices=["music", "speech", "bach"])
    ap.add_argument("--shift-ms", type=float, default=0.0,
                    help="assume the response is this many ms later than the marker (MUSIN-G: 62)")
    ap.add_argument("--zero-shot", default=None,
                    help="evaluate this checkpoint without training")
    ap.add_argument("--model", default="v2", choices=["v1", "v2"])
    ap.add_argument("--feature", default="onset")
    ap.add_argument("--cv", type=int, default=5, help="number of folds")
    ap.add_argument("--cv-by", default="listener", choices=["listener", "song", "both"],
                    help="what is held out: listeners, songs, or both at once (the hardest test)")
    ap.add_argument("--val-groups", type=int, default=2, help="groups kept for early stopping")
    ap.add_argument("--subjects", type=int, default=60, help="speech listeners to use")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--per-rec", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=5e-2)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--channel-drop", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--init", default=None)
    ap.add_argument("--windows", default="5,10,30")
    ap.add_argument("--tag", default="cv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows = [float(w) for w in args.windows.split(",")]
    out = Path("results") / f"cv_{args.tag}.json"
    out.parent.mkdir(exist_ok=True)

    t0 = time.time()
    shift = int(round(args.shift_ms / 1000 * FS))
    if args.dataset == "music":
        recs = load_music(args.feature, shift=shift)
    elif args.dataset == "bach":
        recs = load_bach(args.feature, shift=shift)
    else:
        recs = load_speech(args.feature, args.subjects)
    if shift:
        log.info("applied a %+.0f ms correction (%d samples)", args.shift_ms, shift)
    subs = sorted({r["sub"] for r in recs})
    log.info("%s: %d recordings, %d listeners (%.0fs)", args.dataset, len(recs), len(subs), time.time() - t0)

    if args.zero_shot:
        model = build(args.model, recs[0]["eeg"].shape[0], args.dim, 0.0).to(device)
        model.load_state_dict(torch.load(args.zero_shot, map_location=device)["model"], strict=False)
        res = per_listener(model, recs, windows, device)
        out.write_text(json.dumps({"args": vars(args), "per_listener": res}, indent=1))
        print(f"\n=== {args.tag}: zero-shot {Path(args.zero_shot).name} on {args.dataset} "
              f"(shift {args.shift_ms:+.0f} ms), {len(res)} listeners")
        print(summarise(res, windows))
        print(f"wrote {out}")
        return 0

    if args.dataset == "speech":
        test_s, val_s = set(subs[-4:]), set(subs[-6:-4])
        train = [r for r in recs if r["sub"] not in test_s | val_s]
        val = [r for r in recs if r["sub"] in val_s]
        test = [r for r in recs if r["sub"] in test_s]
        model, best_epoch, hist = train_one(train, val, args, device, rng, args.init)
        ckpt = DATA_ROOT / "models" / f"{args.tag}_best.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "epoch": best_epoch, "args": vars(args)}, ckpt)
        res = per_listener(model, test, windows, device)
        music = per_listener(model, load_music(args.feature), windows, device)
        out.write_text(json.dumps({"args": vars(args), "best_epoch": best_epoch, "history": hist,
                                   "speech_test": res, "music_zero_shot": music}, indent=1))
        print(f"\n=== {args.tag}: speech pretraining, best epoch {best_epoch} (chosen on validation)")
        print("held-out speech listeners:\n" + summarise(res, windows))
        print("music, zero-shot:\n" + summarise(music, windows))
        print(f"checkpoint: {ckpt}")
        return 0

    # Hold out listeners, songs, or both. "song" tests generalisation to music never heard in
    # training; "both" additionally holds out the listeners, which is the hardest setting.
    key = "sub" if args.cv_by == "listener" else "song"
    groups = sorted({r[key] for r in recs})
    folds = np.array_split(np.array(groups, dtype=object), min(args.cv, len(groups)))
    sub_folds = np.array_split(np.array(subs, dtype=object), len(folds)) if args.cv_by == "both" else None
    results, meta = {}, []
    for k, test_g in enumerate(folds):
        test_g = set(test_g)
        rest = [g for g in groups if g not in test_g]
        val_g = {rest[(k * args.val_groups + i) % len(rest)] for i in range(min(args.val_groups, len(rest)))}
        train = [r for r in recs if r[key] not in test_g | val_g]
        val = [r for r in recs if r[key] in val_g]
        test = [r for r in recs if r[key] in test_g]
        if args.cv_by == "both":                     # also hold the listeners out
            test_subs = set(sub_folds[k])
            train = [r for r in train if r["sub"] not in test_subs]
            val = [r for r in val if r["sub"] not in test_subs]
            test = [r for r in test if r["sub"] in test_subs]
        if not test or not train:
            log.warning("fold %d empty, skipped", k + 1)
            continue
        log.info("fold %d/%d  held-out %s %s  val %s  train %d recordings  test %d", k + 1, len(folds),
                 key, sorted(test_g), sorted(val_g), len(train), len(test))
        model, best_epoch, hist = train_one(train, val, args, device, rng, args.init)
        fold_res = per_listener(model, test, windows, device)
        meta.append({"fold": k, "held_out": sorted(map(str, test_g)), "val": sorted(map(str, val_g)),
                     "cv_by": args.cv_by, "best_epoch": best_epoch, "history": hist})
        for s, r in fold_res.items():
            log.info("  test %s %s", s, {kk: round(v, 3) for kk, v in r.items()})
        for s, r in fold_res.items():          # listeners recur across song folds: average them
            if s in results:
                results[s] = {kk: (results[s][kk] + r[kk]) / 2 for kk in r}
            else:
                results[s] = r
        out.write_text(json.dumps({"args": vars(args), "folds": meta, "per_listener": results}, indent=1))

    print(f"\n=== {args.tag}: {args.model}, {len(folds)}-fold CV held out by {args.cv_by}, "
          f"{len(results)} listeners")
    print(summarise(results, windows))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
