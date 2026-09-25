"""Speech-initialised music decoding: ridge regression with the speech weights as the prior.

Ordinary ridge shrinks the solution toward zero. Here it is shrunk toward a decoder trained on
speech listening, which is what fine-tuning means for a linear model:

    w = argmin ||Xw - y||^2 + lam * ||w - w_speech||^2
      = (X'X + lam I)^-1 (X'y + lam * w_speech)

Sweeping `lam` traces the whole path: lam -> 0 is music-only (Gate A), lam -> infinity is the
zero-shot speech decoder. Evaluation is leave-one-music-listener-out throughout, so the held-out
listener never contributes to either the prior or the music fit.

    python -m eg606.eval.transfer_prior --speech-subjects 20 --windows 10,30
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from eg606.eval.linear import FS, accumulate, match_mismatch, predict, zscore
from eg606.paths import derived_dir

log = logging.getLogger(__name__)
PRIOR_LAMBDAS = np.logspace(3, 10, 15)


def solve_with_prior(xtx, xty, lam, prior=None):
    rhs = xty if prior is None else xty + lam * prior
    return cho_solve(cho_factor(xtx + lam * np.eye(len(xtx))), rhs)


def speech_stats(root: Path, feats: dict, kind: str, n_subjects: int):
    xtx = xty = None
    for p in sorted(root.glob("sub-*.npz"))[:n_subjects]:
        d = np.load(p, allow_pickle=True)
        for m in json.loads(str(d["meta"])):
            y = feats.get(f"{kind}::{m['stimulus']}")
            if y is None:
                continue
            eeg = zscore(d[m["key"]].astype(np.float64))
            n = min(eeg.shape[1], len(y))
            xtx, xty = accumulate(eeg[:, :n], zscore(y[:n].astype(np.float64)), xtx, xty)
        log.info("speech %s accumulated", p.stem)
    return xtx, xty


def music_stats(root: Path, feats: dict, kind: str, variant: str):
    pattern = re.compile(rf"^sub-\d+_{re.escape(variant)}$")
    out = {}
    for p in sorted(x for x in root.glob("sub-*.npz") if pattern.match(x.stem)):
        d = np.load(p, allow_pickle=True)
        xtx = xty = None
        trials = []
        for m in json.loads(str(d["meta"])):
            song = m["song"]
            eeg = zscore(d[f"song{song:02d}"].astype(np.float64))
            y = feats[f"{kind}{song:02d}"].astype(np.float64)
            n = min(eeg.shape[1], len(y))
            y = zscore(y[:n])
            xtx, xty = accumulate(eeg[:, :n], y, xtx, xty)
            trials.append((eeg[:, :n], y))
        out[p.stem.split("_")[0]] = {"xtx": xtx, "xty": xty, "trials": trials}
        log.info("music %s accumulated", p.stem)
    return out


def score(trials, w, windows):
    rs, mm = [], {f"{x:g}s": [0, 0] for x in windows}
    for eeg, y in trials:
        pred = predict(eeg, w, len(y))
        if pred.std() == 0:
            continue
        rs.append(np.corrcoef(pred, y)[0, 1])
        for x in windows:
            h, t = match_mismatch(pred, y, int(x * FS))
            mm[f"{x:g}s"][0] += h
            mm[f"{x:g}s"][1] += t
    return (float(np.mean(rs)) if rs else float("nan"),
            {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in mm.items()})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.eval.transfer_prior")
    ap.add_argument("--speech-subjects", type=int, default=20)
    ap.add_argument("--speech-lambda", type=float, default=1e7, help="ridge for the speech prior")
    ap.add_argument("--variant", default="bs64", help="music variant sharing the speech montage")
    ap.add_argument("--feature", default="env", choices=["env", "onset"])
    ap.add_argument("--windows", default="10,30")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    windows = [float(w) for w in args.windows.split(",")]

    sp_root, mu_root = derived_dir("sparrkulee"), derived_dir("musin_g")
    sp_feats = dict(np.load(sp_root / "audio_features.npz"))
    mu_feats = dict(np.load(mu_root / "audio_features.npz"))

    sxtx, sxty = speech_stats(sp_root, sp_feats, args.feature, args.speech_subjects)
    w_speech = solve_with_prior(sxtx, sxty, args.speech_lambda)
    log.info("speech prior ready (%d features)", len(w_speech))

    music = music_stats(mu_root, mu_feats, args.feature, args.variant)
    subs = sorted(music)
    log.info("%d music listeners", len(subs))

    results = {"lambdas": PRIOR_LAMBDAS.tolist(), "with_prior": {}, "no_prior": {}}
    for held in subs:
        others = [s for s in subs if s != held]
        xtx = sum(music[s]["xtx"] for s in others)
        xty = sum(music[s]["xty"] for s in others)
        for tag, prior in (("with_prior", w_speech), ("no_prior", None)):
            per_lam = []
            for lam in PRIOR_LAMBDAS:
                w = solve_with_prior(xtx, xty, lam, prior)
                r, mm = score(music[held]["trials"], w, windows)
                per_lam.append({"r": r, "mm": mm})
            results[tag][held] = per_lam
        best = max(range(len(PRIOR_LAMBDAS)),
                   key=lambda i: results["with_prior"][held][i]["mm"][f"{windows[-1]:g}s"])
        log.info("%s  best-lam=%.0e  with_prior=%.3f  no_prior=%.3f (%gs)", held,
                 PRIOR_LAMBDAS[best],
                 results["with_prior"][held][best]["mm"][f"{windows[-1]:g}s"],
                 results["no_prior"][held][best]["mm"][f"{windows[-1]:g}s"], windows[-1])

    print("\n=== speech prior vs music-only, leave-one-music-listener-out (chance = 0.5)")
    for x in windows:
        k = f"{x:g}s"
        print(f"\nwindow {k}")
        print(f"{'lambda':>10} {'music-only':>12} {'speech-prior':>14}  (mean over listeners)")
        for i, lam in enumerate(PRIOR_LAMBDAS):
            a = np.mean([results["no_prior"][s][i]["mm"][k] for s in subs])
            b = np.mean([results["with_prior"][s][i]["mm"][k] for s in subs])
            mark = "  <-- prior helps" if b > a + 0.005 else ""
            print(f"{lam:10.0e} {a:12.3f} {b:14.3f}{mark}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
