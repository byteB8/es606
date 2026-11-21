"""Classifier baselines for naming the song a new listener hears, on the decoder's own decisions.

The published approach to EEG song identification trains a classifier to map short EEG windows to
the song label: GuessTheMusic uses a spectrogram CNN, and logistic regression on band power is the
simplest member of the family. For a fair comparison with the decoder in eval/newlistener.py,
each classifier gets exactly the same test decisions:

  * train on every window of the other 19 listeners (whole trials, as the decoder does);
  * for the held-out listener, cut the first n_common samples of each trial into the same
    non-overlapping decision windows (2 s ... "all");
  * inside a decision window, add the classifier's log-probabilities over its short windows (2 s
    for logistic regression, 1 s for the CNN), i.e. treat them as independent evidence.

Chance is 1/12 throughout.

    python -m eg606.eval.newlistener_baselines --model logreg --out results/newlistener/logreg_basic.json
    python -m eg606.eval.newlistener_baselines --model cnn --out results/newlistener/cnn_basic.json
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

from eg606.eval.newlistener import print_table, summarize
from eg606.paths import derived_dir

log = logging.getLogger(__name__)


def load(variant: str, max_subjects=None):
    """(sub, song, eeg) per trial, and n_common computed exactly as the decoder computes it."""
    root = derived_dir("musin_g")
    feats = np.load(root / "audio_features.npz")
    suffix = "" if variant == "none" else f"_{variant}"
    pattern = re.compile(rf"^sub-\d+{re.escape(suffix)}$")
    paths = sorted(p for p in root.glob("sub-*.npz") if pattern.match(p.stem))
    paths = paths[:max_subjects] if max_subjects else paths
    trials, fs = [], None
    for p in paths:
        d = np.load(p, allow_pickle=True)
        fs = float(d["fs"])
        if fs != 64.0:
            raise ValueError("the classifier features are defined at 64 Hz")
        for m in json.loads(str(d["meta"])):
            song = m["song"]
            eeg = d[f"song{song:02d}"].astype(np.float64)
            n = min(eeg.shape[1], len(feats[f"onset{song:02d}"]))
            trials.append({"sub": p.stem.split("_")[0], "song": song, "eeg": eeg, "n": n})
    return trials, fs, min(t["n"] for t in trials)


def featurize(model: str, eeg: np.ndarray) -> np.ndarray:
    if model == "logreg":
        from eg606.eval.songid import band_features
        return band_features(eeg)                       # 2 s windows
    from eg606.models.cnn_songid import spectrograms
    return spectrograms(eeg)                            # 1 s windows


def short_window(model: str) -> int:
    from eg606.eval.songid import WIN
    from eg606.models.cnn_songid import FS, WIN_S
    return WIN if model == "logreg" else int(WIN_S * FS)


def decisions(logp: np.ndarray, short: int, win: int, n_common: int):
    """Sum log-probabilities of the short windows inside each decision window."""
    out = []
    for a in range(0, n_common - win + 1, win):
        lo, hi = -(-a // short), (a + win) // short      # short windows fully inside [a, a+win)
        if hi > lo:
            out.append(logp[lo:hi].sum(0))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.eval.newlistener_baselines")
    ap.add_argument("--model", required=True, choices=["logreg", "cnn"])
    ap.add_argument("--variant", default="basic")
    ap.add_argument("--windows", default="2,4,10,20,30,50,all")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--max-subjects", type=int, default=None, help="smoke tests only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    trials, fs, n_common = load(args.variant, args.max_subjects)
    short = short_window(args.model)
    songs = sorted({t["song"] for t in trials})
    subs = sorted({t["sub"] for t in trials})
    wins = {}
    for w in args.windows.split(","):
        n = n_common if w == "all" else int(round(float(w) * fs))
        if n <= n_common and n >= short and (w == "all" or n % short == 0):
            wins[w] = n
        else:
            log.info("window %s skipped: not a multiple of the %d-sample classifier window", w, short)
    for t in trials:
        t["X"] = featurize(args.model, t["eeg"])
        t["X_test"] = t["X"][: n_common // short]        # windows inside the first n_common samples
        del t["eeg"]
    log.info("%s: %d trials, %d listeners, feature shape %s, n_common %d, windows %s",
             args.model, len(trials), len(subs), trials[0]["X"].shape[1:], n_common, list(wins))

    device = "cpu"
    if args.model == "cnn":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("device %s", device)

    t0, rec = time.time(), {}
    for L in subs:
        train = [t for t in trials if t["sub"] != L]
        Xtr = np.concatenate([t["X"] for t in train])
        ytr = np.concatenate([np.full(len(t["X"]), songs.index(t["song"])) for t in train])
        if args.model == "logreg":
            from eg606.eval.songid import clf
            model = clf().fit(Xtr, ytr)
            classes = list(model.classes_)
            if classes != list(range(len(songs))):
                raise RuntimeError("a training split is missing a song")
            log_proba = model.predict_log_proba
        else:
            from eg606.models.cnn_songid import fit
            log_proba = fit(Xtr, ytr, len(songs), device, args.epochs, args.seed)
        slot = rec[L] = {"short_acc": [], "win": {w: {"real": [0, 0]} for w in wins},
                         "guess_all": {}}
        for t in (t for t in trials if t["sub"] == L):
            logp = log_proba(t["X_test"])
            label = songs.index(t["song"])
            slot["short_acc"].append(float((logp.argmax(1) == label).mean()))
            for w, n in wins.items():
                for d in decisions(logp, short, n, n_common):
                    g = int(np.argmax(d))
                    slot["win"][w]["real"][0] += int(g == label)
                    slot["win"][w]["real"][1] += 1
                    if w == "all":
                        slot["guess_all"][str(t["song"])] = songs[g]
        log.info("%s %s short-window acc %.3f, whole-song hits %s (%.0f s)", args.model, L,
                 np.mean(slot["short_acc"]), slot["win"].get("all", {}).get("real"),
                 time.time() - t0)

    summary = summarize(rec, len(songs), conds=("real",))
    res = {"model": args.model, "variant": args.variant, "fs": fs, "n_common": n_common,
           "n_common_s": n_common / fs, "short_window": short, "songs": songs, "windows": wins,
           "seed": args.seed, "per_listener": rec, "summary": summary}
    print_table(f"{args.model} baseline, new listener song ID, chance 1/12 = 0.083",
                {args.model: summary}, conds=("real",))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1, default=float))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
