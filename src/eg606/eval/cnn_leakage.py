"""The leakage audit again, with the published family of model instead of a linear one.

Same splits and same data as eval/leakage.py; only the classifier changes. The question is narrow:
does a convolutional network trained on spectrograms also identify songs from the silence before
they start, and does it also collapse on a listener it has not seen? If it does, the protocol is
the problem. If instead it generalises across listeners where the linear model could not, that is
worth knowing too, and this reports it either way.

    python train.py --config configs/cnn_musin_g.json     # generic name, see scripts/
    python -m eg606.eval.cnn_leakage --dataset musin_g --out results/e3/cnn_musin_g.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from eg606.eval.leakage import Windows, _stack
from eg606.models.cnn_songid import SongCNN, spectrograms
from eg606.paths import derived_dir

log = logging.getLogger(__name__)
TRAIN_FRACTION = 0.7


def load_musin_g(variant: str = "basic"):
    root = derived_dir("musin_g")
    suffix = "" if variant == "none" else f"_{variant}"
    pattern = re.compile(rf"^sub-\d+{re.escape(suffix)}$")
    music, silence = [], []
    for p in sorted(x for x in root.glob("sub-*.npz") if pattern.match(x.stem)):
        d = np.load(p, allow_pickle=True)
        sub = p.stem.split("_")[0]
        for m in json.loads(str(d["meta"])):
            song = m["song"]
            for f in spectrograms(d[f"song{song:02d}"].astype(np.float64)):
                music.append((sub, str(song), song, "a", f))
            for f in spectrograms(d[f"pre{song:02d}"].astype(np.float64)):
                silence.append((sub, str(song), song, "a", f))
    return _stack(music), (_stack(silence) if silence else None)


def load_nmed(name: str):
    root = derived_dir(name)
    music, silence = [], []
    for p in sorted(root.glob("sub-*.npz")):
        is_silence = p.stem.endswith("_silence")
        sub = p.stem.replace("_silence", "")
        d = np.load(p, allow_pickle=True)
        for m in json.loads(str(d["meta"])):
            label = m.get("stimulus") or str(m["song"])
            for f in spectrograms(d[m["key"]].astype(np.float64)):
                (silence if is_silence else music).append(
                    (sub, label, m["song"], m.get("rep") or "a", f))
    return _stack(music), (_stack(silence) if silence else None)


def train_eval(w: Windows, tr, te, classes=None, device="cpu", epochs=25, seed=0) -> float | None:
    """Fit the CNN on tr, return accuracy on te. Standardised per feature on the training split.

    The label set is whatever the *training* split contains, not every label in the dataset. On
    NMED-H a listener heard four of the sixteen stimuli, so scoring them against all sixteen would
    make the task harder than the one the linear audit ran and the two would not be comparable.
    """
    torch.manual_seed(seed)
    classes = sorted(set(w.label[tr])) if classes is None else classes
    index = {c: i for i, c in enumerate(classes)}
    ytr = np.array([index[c] for c in w.label[tr]])
    yte = np.array([index.get(c, -1) for c in w.label[te]])
    if len(set(ytr)) < 2 or not len(yte):
        return None
    Xtr, Xte = w.X[tr].astype(np.float32), w.X[te].astype(np.float32)
    mu, sd = Xtr.mean((0, 3), keepdims=True), Xtr.std((0, 3), keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    model = SongCNN(Xtr.shape[1], len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    lossf = nn.CrossEntropyLoss()
    xtr = torch.from_numpy(Xtr)
    ttr = torch.from_numpy(ytr).long()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            loss = lossf(model(xtr[b].to(device)), ttr[b].to(device))
            loss.backward()
            opt.step()
    model.eval()
    preds = []
    with torch.no_grad():
        xte = torch.from_numpy(Xte)
        for i in range(0, len(xte), 256):
            preds.append(model(xte[i:i + 256].to(device)).argmax(1).cpu().numpy())
    return float((np.concatenate(preds) == yte).mean())


def conditions(music: Windows, silence: Windows | None, device, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}

    def within(w, key):
        accs = {}
        for s in sorted(set(w.subject)):
            idx = np.where(w.subject == s)[0]
            rng.shuffle(idx)
            cut = int(len(idx) * TRAIN_FRACTION)
            a = train_eval(w, idx[:cut], idx[cut:], None, device, seed=seed)
            if a is not None:
                accs[s] = a
                log.info("  %s %s %.3f", key, s, a)
        out[key] = accs

    def loso(w, key):
        cls = sorted(set(w.label))
        accs = {}
        for s in sorted(set(w.subject)):
            a = train_eval(w, np.where(w.subject != s)[0], np.where(w.subject == s)[0],
                           cls, device, seed=seed)
            if a is not None:
                accs[s] = a
                log.info("  %s %s %.3f", key, s, a)
        out[key] = accs

    within(music, "within_naive")
    loso(music, "loso")
    if silence is not None:
        within(silence, "silence_naive")
        loso(silence, "silence_loso")
    if len(set(music.rep)) > 1:
        accs = {}
        for s in sorted(set(music.subject)):
            mine = music.subject == s
            tr = np.where(mine & (music.rep == "a"))[0]
            te = np.where(mine & (music.rep == "b"))[0]
            if len(tr) and len(te):
                a = train_eval(music, tr, te, None, device, seed=seed)
                if a is not None:
                    accs[s] = a
        out["cross_repetition"] = accs
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.eval.cnn_leakage")
    ap.add_argument("--dataset", default="musin_g")
    ap.add_argument("--variant", default="basic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device %s", device)

    music, silence = (load_musin_g(args.variant) if args.dataset == "musin_g"
                      else load_nmed(args.dataset))
    log.info("%d music windows %s, %d labels, %d listeners", len(music.label), music.X.shape[1:],
             len(set(music.label)), len(set(music.subject)))
    res = {"dataset": args.dataset, "seed": args.seed,
           "chance": music.chance(), "n_labels": len(set(music.label)),
           "chance_per_listener": {s: music.chance(s) for s in sorted(set(music.subject))},
           "conditions": conditions(music, silence, device, args.seed)}

    print(f"\n=== spectrogram CNN, {args.dataset}, chance {res['chance']:.3f}")
    print(f"  {'condition':<24}{'mean':>8}{'sd':>8}{'x chance':>10}")
    ch = res["chance_per_listener"]
    for cond, accs in res["conditions"].items():
        a = np.array(list(accs.values()))
        if len(a):
            base = ch if cond.startswith(("within", "cross")) else None
            rel = (np.mean([accs[s] / ch[s] for s in accs]) if base
                   else a.mean() / res["chance"])
            print(f"  {cond:<24}{a.mean():8.3f}{a.std():8.3f}{rel:9.1f}x")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
