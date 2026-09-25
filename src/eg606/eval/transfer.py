"""Does a decoder trained on speech listening work on music listening?

Trains one linear backward model on SparrKULee speech and applies it, with no music training at
all, to MUSIN-G listeners interpolated onto the speech montage. Reference points printed alongside:
speech->speech on held-out speech listeners (the decoder works at all) and the Gate A music->music
number (what training on music buys).

    python -m eg606.eval.transfer --speech-subjects 20 --windows 5,10,30,60
"""
from __future__ import annotations

import argparse
import json
import re
import logging
import sys
from pathlib import Path

import numpy as np

from eg606.eval.linear import FS, accumulate, match_mismatch, predict, solve, zscore
from eg606.eval.protocol import bandpass
from eg606.paths import derived_dir

log = logging.getLogger(__name__)
LAMBDAS = np.logspace(2, 11, 19)


def speech_trials(path: Path, feats: dict, kind: str, band=None):
    d = np.load(path, allow_pickle=True)
    for m in json.loads(str(d["meta"])):
        eeg = d[m["key"]].astype(np.float64)
        y = feats.get(f"{kind}::{m['stimulus']}")
        if y is None:
            continue
        n = min(eeg.shape[1], len(y))
        eeg, y = eeg[:, :n], y[:n].astype(np.float64)
        if band:
            eeg, y = bandpass(eeg, *band), bandpass(y[None], *band)[0]
        yield zscore(eeg), zscore(y)


def music_trials(path: Path, feats: dict, kind: str, band=None):
    d = np.load(path, allow_pickle=True)
    for m in json.loads(str(d["meta"])):
        song = m["song"]
        eeg = d[f"song{song:02d}"].astype(np.float64)
        y = feats[f"{kind}{song:02d}"].astype(np.float64)
        n = min(eeg.shape[1], len(y))
        eeg, y = eeg[:, :n], y[:n]
        if band:
            eeg, y = bandpass(eeg, *band), bandpass(y[None], *band)[0]
        yield song, zscore(eeg), zscore(y)


def evaluate(trials, w, windows) -> dict:
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
    return {"r": float(np.mean(rs)) if rs else float("nan"),
            "mm": {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in mm.items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.eval.transfer")
    ap.add_argument("--speech-subjects", type=int, default=20, help="how many listeners to train on")
    ap.add_argument("--holdout", type=int, default=4, help="speech listeners kept for validation")
    ap.add_argument("--feature", default="env", choices=["env", "onset"])
    ap.add_argument("--windows", default="5,10,30,60")
    ap.add_argument("--band", default=None, help="e.g. 4,8 to restrict EEG and target to 4-8 Hz")
    ap.add_argument("--music-file", default="{sub}_bs64.npz",
                    help="music file pattern (must share the speech montage)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    windows = [float(w) for w in args.windows.split(",")]
    band = tuple(float(x) for x in args.band.split(",")) if args.band else None

    sp_root, mu_root = derived_dir("sparrkulee"), derived_dir("musin_g")
    sp_feats = dict(np.load(sp_root / "audio_features.npz"))
    mu_feats = dict(np.load(mu_root / "audio_features.npz"))

    sp_subs = sorted(sp_root.glob("sub-*.npz"))[: args.speech_subjects + args.holdout]
    if len(sp_subs) < args.speech_subjects + args.holdout:
        log.warning("only %d speech listeners available", len(sp_subs))
    train_subs, val_subs = sp_subs[: args.speech_subjects], sp_subs[args.speech_subjects:]

    xtx = xty = None
    for p in train_subs:
        for eeg, y in speech_trials(p, sp_feats, args.feature, band):
            xtx, xty = accumulate(eeg, y, xtx, xty)
        log.info("accumulated %s", p.stem)
    if xtx is None:
        log.error("no speech training data found"); return 1

    best_lam, best_r = None, -np.inf
    for lam in LAMBDAS:
        w = solve(xtx, xty, lam)
        rs = [np.corrcoef(predict(e, w, len(y)), y)[0, 1]
              for p in val_subs for e, y in speech_trials(p, sp_feats, args.feature, band)]
        if rs and np.mean(rs) > best_r:
            best_lam, best_r = lam, float(np.mean(rs))
    log.info("ridge lambda=%.0e (held-out speech r=%.4f)", best_lam, best_r)
    w = solve(xtx, xty, best_lam)

    results = {"lambda": float(best_lam), "speech_holdout_r": best_r, "band": band,
               "speech": {}, "music": {}}
    for p in val_subs:
        results["speech"][p.stem] = evaluate(speech_trials(p, sp_feats, args.feature, band), w, windows)
        log.info("speech %s  r=%.4f  %s", p.stem, results["speech"][p.stem]["r"],
                 {k: round(v, 3) for k, v in results["speech"][p.stem]["mm"].items()})

    # strict match: a loose glob also picks up e.g. sub-001_basic_bs64 and double-counts listeners
    strict = re.compile("^" + re.escape(args.music_file.format(sub="SUB")).replace("SUB", r"sub-\d+") + "$")
    for p in sorted(x for x in mu_root.glob(args.music_file.format(sub="sub-*")) if strict.match(x.name)):
        sub = p.stem.split("_")[0]
        trials = [(e, y) for _, e, y in music_trials(p, mu_feats, args.feature, band)]
        results["music"][sub] = evaluate(iter(trials), w, windows)
        log.info("music  %s  r=%.4f  %s", sub, results["music"][sub]["r"],
                 {k: round(v, 3) for k, v in results["music"][sub]["mm"].items()})

    print(f"\n=== speech-trained decoder, no music training, band={band} (chance = 0.5)")
    for domain in ("speech", "music"):
        if not results[domain]:
            continue
        r = np.array([v["r"] for v in results[domain].values()])
        print(f"{domain:6}: n={len(r)}  envelope r {r.mean():+.4f}")
        for x in windows:
            k = f"{x:g}s"
            a = np.array([v["mm"][k] for v in results[domain].values()])
            print(f"          {k:>4}: {a.mean():.3f} +/- {a.std():.3f}  "
                  f"({(a > 0.5).sum()}/{len(a)} above chance)")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
