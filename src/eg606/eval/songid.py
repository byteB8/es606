"""Song identification under the published protocol vs held-out listeners: the leakage figure.

Reproduces the GuessTheMusic-style setup — spectral band-power features from short windows and a
12-way song classifier — under five conditions:

  within_naive      per listener, random 70/30 windows (the published within-subject protocol)
  pooled_naive      all listeners pooled, random 70/30 windows
  loso              train on 19 listeners, test on the held-out one
  silence_naive     within_naive, but on the 10 s of SILENCE before each song (label = upcoming song)
  silence_loso      loso on the silence windows

The silence conditions are the control that matters: there is no music in those windows, so any
above-chance accuracy there is block identity, not music. Chance is 1/12 = 0.083.

    python -m eg606.eval.songid --variant basic
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eg606.paths import derived_dir

log = logging.getLogger(__name__)
FS = 64
WIN = 2 * FS
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
TRAIN_FRACTION = 0.7


def band_features(eeg: np.ndarray) -> np.ndarray:
    """(channels, time) -> (windows, channels * bands) log band power, non-overlapping 2 s windows."""
    c, t = eeg.shape
    n = t // WIN
    if n == 0:
        return np.zeros((0, c * len(BANDS)))
    x = eeg[:, : n * WIN].reshape(c, n, WIN).transpose(1, 0, 2)          # (n, c, WIN)
    spec = np.abs(np.fft.rfft(x * np.hanning(WIN), axis=-1)) ** 2
    freqs = np.fft.rfftfreq(WIN, 1 / FS)
    feats = [np.log(spec[..., (freqs >= lo) & (freqs < hi)].mean(-1) + 1e-20) for lo, hi in BANDS.values()]
    return np.concatenate(feats, axis=1)                                  # (n, c*bands)


def load(variant: str):
    root = derived_dir("musin_g")
    suffix = "" if variant == "none" else f"_{variant}"
    pattern = re.compile(rf"^sub-\d+{re.escape(suffix)}$")
    music, silence = [], []
    for p in sorted(x for x in root.glob("sub-*.npz") if pattern.match(x.stem)):
        d = np.load(p, allow_pickle=True)
        sub = p.stem.split("_")[0]
        for m in json.loads(str(d["meta"])):
            song = m["song"]
            for f in band_features(d[f"song{song:02d}"].astype(np.float64)):
                music.append((sub, song, f))
            for f in band_features(d[f"pre{song:02d}"].astype(np.float64)):
                silence.append((sub, song, f))
    return music, silence


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))


def as_arrays(rows):
    return (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
            np.stack([r[2] for r in rows]))


def within_naive(rows, rng, shuffle=False):
    subs, y, X = as_arrays(rows)
    accs = {}
    for s in sorted(set(subs)):
        idx = np.where(subs == s)[0]
        rng.shuffle(idx)
        cut = int(len(idx) * TRAIN_FRACTION)
        tr, te = idx[:cut], idx[cut:]
        ytr = rng.permutation(y[tr]) if shuffle else y[tr]
        if len(set(ytr)) < 2:
            continue
        accs[s] = float((clf().fit(X[tr], ytr).predict(X[te]) == y[te]).mean())
    return accs


def pooled_naive(rows, rng):
    subs, y, X = as_arrays(rows)
    idx = rng.permutation(len(y))
    cut = int(len(idx) * TRAIN_FRACTION)
    model = clf().fit(X[idx[:cut]], y[idx[:cut]])
    te = idx[cut:]
    pred = model.predict(X[te])
    return {s: float((pred[subs[te] == s] == y[te][subs[te] == s]).mean())
            for s in sorted(set(subs[te]))}


def loso(rows):
    subs, y, X = as_arrays(rows)
    accs = {}
    for s in sorted(set(subs)):
        tr, te = subs != s, subs == s
        accs[s] = float((clf().fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean())
        log.info("loso %s %.3f", s, accs[s])
    return accs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.eval.songid")
    ap.add_argument("--variant", default="basic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(args.seed)

    music, silence = load(args.variant)
    log.info("%d music windows, %d silence windows, %d features",
             len(music), len(silence), len(music[0][2]))

    conds = {
        "within_naive": lambda: within_naive(music, rng),
        "within_naive_shuffled": lambda: within_naive(music, rng, shuffle=True),
        "pooled_naive": lambda: pooled_naive(music, rng),
        "loso": lambda: loso(music),
        "silence_naive": lambda: within_naive(silence, rng),
        "silence_loso": lambda: loso(silence),
    }
    results = {}
    for name, fn in conds.items():
        results[name] = fn()
        a = np.array(list(results[name].values()))
        log.info("%-22s mean %.3f", name, a.mean())

    print(f"\n=== song identification, 12-way (chance = 0.083), variant={args.variant}")
    print(f"{'condition':<24}{'mean':>7}{'sd':>7}  listeners above chance")
    for name, accs in results.items():
        a = np.array(list(accs.values()))
        print(f"{name:<24}{a.mean():7.3f}{a.std():7.3f}  {(a > 1/12).sum()}/{len(a)}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
