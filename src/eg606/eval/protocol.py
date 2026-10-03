"""Compare evaluation protocols on the same data, model and feature — the leakage measurement.

  naive  train and test windows come from the SAME recordings (the published protocol)
  loso   leave one listener out  -> new listener, seen songs
  song   leave one song out      -> new song, seen listeners

Only the split changes, so the differences are the protocol, not the model. `--band` restricts both
EEG and target to a frequency band, which is how we ask *where* the shared signal lives (RQ2).

Speed: X'X is accumulated once per listener and once per song, so every fold is a matrix sum plus a
Cholesky solve instead of a refit.

    python -m eg606.eval.protocol --variant none --feature onset --splits naive,loso,song
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from eg606.eval.linear import (FS, accumulate, deranged, match_mismatch, predict,
                               solve, zscore)
from eg606.paths import derived_dir

log = logging.getLogger(__name__)
LAMBDAS = np.logspace(3, 9, 13)
CHUNK_S = 5.0
TRAIN_FRACTION = 0.7


def bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    b, a = butter(4, [lo, min(hi, FS / 2 - 0.5)], btype="band", fs=FS)
    return filtfilt(b, a, x, axis=-1)


def load_trials(variant: str, feature: str, band=None) -> list[dict]:
    root = derived_dir("musin_g")
    feats = dict(np.load(root / "audio_features.npz"))
    suffix = "" if variant == "none" else f"_{variant}"
    pattern = re.compile(rf"^sub-\d+{re.escape(suffix)}$")
    trials = []
    for p in sorted(x for x in root.glob("sub-*.npz") if pattern.match(x.stem)):
        d = np.load(p, allow_pickle=True)
        sub = p.stem.split("_")[0]
        for m in json.loads(str(d["meta"])):
            song = m["song"]
            eeg = d[f"song{song:02d}"].astype(np.float64)
            y = feats[f"{feature}{song:02d}"].astype(np.float64)
            n = min(eeg.shape[1], len(y))
            eeg, y = eeg[:, :n], y[:n]
            if band:
                eeg, y = bandpass(eeg, *band), bandpass(y[None], *band)[0]
            trials.append({"sub": sub, "song": song, "eeg": zscore(eeg), "y": zscore(y)})
    return trials


def group_stats(trials, key: str):
    """X'X and X'y accumulated once per group (listener or song)."""
    out = {}
    for t in trials:
        g = t[key]
        xtx, xty = out.get(g, (None, None))
        out[g] = accumulate(t["eeg"], t["y"], xtx, xty)
    return out


def score(segments, w, windows):
    rs, mm = [], {f"{x:g}s": [0, 0] for x in windows}
    for eeg, y in segments:
        if len(y) < 64:
            continue
        pred = predict(eeg, w, len(y))
        if pred.std() == 0 or y.std() == 0:
            continue
        rs.append(np.corrcoef(pred, y)[0, 1])
        for x in windows:
            h, t = match_mismatch(pred, y, int(x * FS))
            mm[f"{x:g}s"][0] += h
            mm[f"{x:g}s"][1] += t
    return (float(np.mean(rs)) if rs else float("nan"),
            {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in mm.items()})


def pick_lambda(xtx, xty, val_segments, windows):
    best, best_r = LAMBDAS[len(LAMBDAS) // 2], -np.inf
    for lam in LAMBDAS:
        r, _ = score(val_segments, solve(xtx, xty, lam), windows)
        if np.isfinite(r) and r > best_r:
            best, best_r = lam, r
    return best


def held_out_split(trials, split, groups, held):
    """(training trials, test trials) for one fold of a group-wise split."""
    key = "sub" if split == "loso" else "song"
    train = [t for t in trials if t[key] != held]
    test = [t for t in trials if t[key] == held]
    return train, test


def run_group_split(split, trials, stats, windows, rng=None):
    key = "sub" if split == "loso" else "song"
    groups = sorted(stats)
    results, controls = {}, {}
    for held in groups:
        others = [g for g in groups if g != held]
        val_g, fit_g = others[:3], others[3:]
        xtx = sum(stats[g][0] for g in fit_g)
        xty = sum(stats[g][1] for g in fit_g)
        val_seg = [(t["eeg"], t["y"]) for t in trials if t[key] in val_g][:20]
        lam = pick_lambda(xtx, xty, val_seg, windows)

        xtx = sum(stats[g][0] for g in others)
        xty = sum(stats[g][1] for g in others)
        w = solve(xtx, xty, lam)
        test = [(t["eeg"], t["y"]) for t in trials if t[key] == held]
        results[str(held)] = score(test, w, windows)
        log.info("%s %-10s r=%+.4f %s (lam %.0e)", split, held, results[str(held)][0],
                 {k: round(v, 3) for k, v in results[str(held)][1].items()}, lam)
        if rng is not None:
            # a held-out song gives every test segment the same audio, so the swap must come
            # from elsewhere; a held-out listener already spans songs
            alt = ([t["y"] for t in trials if t[key] != held][:200]
                   if key == "song" else None)
            ctrl = deranged(test, rng, alt)
            if ctrl:
                controls[str(held)] = score(ctrl, w, windows)
                log.info("%s %-10s CONTROL %s", split, held,
                         {k: round(v, 3) for k, v in controls[str(held)][1].items()})
    return (results, controls) if rng is not None else results


def run_naive(trials, windows, rng):
    """Train and test windows drawn from the same recordings: reproduces the published protocol."""
    size = int(CHUNK_S * FS)
    xtx = xty = None
    test_by_sub: dict[str, list] = {}
    for t in trials:
        n = t["eeg"].shape[1]
        bounds = [(s, s + size) for s in range(0, n - size + 1, size)]
        train_mask = rng.random(len(bounds)) < TRAIN_FRACTION
        for (a, b), is_train in zip(bounds, train_mask):
            seg = (t["eeg"][:, a:b], t["y"][a:b])
            if is_train:
                xtx, xty = accumulate(seg[0], seg[1], xtx, xty)
            else:
                test_by_sub.setdefault(t["sub"], []).append(seg)
    some = next(iter(test_by_sub.values()))[:20]
    lam = pick_lambda(xtx, xty, some, windows)
    w = solve(xtx, xty, lam)
    results = {sub: score(segs, w, windows) for sub, segs in sorted(test_by_sub.items())}
    for sub, (r, mm) in results.items():
        log.info("naive %-10s r=%+.4f %s", sub, r, {k: round(v, 3) for k, v in mm.items()})
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.eval.protocol")
    ap.add_argument("--variant", default="none")
    ap.add_argument("--feature", default="onset", choices=["env", "onset"])
    ap.add_argument("--splits", default="naive,loso,song")
    ap.add_argument("--windows", default="5,10,30")
    ap.add_argument("--band", default=None, help="e.g. 4,8 to restrict EEG and target to 4-8 Hz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control", action="store_true",
                    help="also score with each target paired to another segment's EEG")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    windows = [float(w) for w in args.windows.split(",")]
    band = tuple(float(x) for x in args.band.split(",")) if args.band else None
    rng = np.random.default_rng(args.seed)
    ctl_rng = np.random.default_rng(args.seed + 1) if args.control else None

    trials = load_trials(args.variant, args.feature, band)
    log.info("%d trials, %d channels, band=%s", len(trials), trials[0]["eeg"].shape[0], band)

    splits = args.splits.split(",")
    summary = {"variant": args.variant, "feature": args.feature, "band": band, "splits": {}}
    by_sub = group_stats(trials, "sub") if "loso" in splits else None
    by_song = group_stats(trials, "song") if "song" in splits else None
    log.info("group statistics ready")

    for split in splits:
        if split == "naive":
            res = run_naive(trials, windows, rng)
        elif split == "loso":
            res = run_group_split("loso", trials, by_sub, windows, ctl_rng)
        elif split == "song":
            res = run_group_split("song", trials, by_song, windows, ctl_rng)
        else:
            ap.error(f"unknown split {split}")
        ctrl = None
        if ctl_rng is not None and isinstance(res, tuple):
            res, ctrl = res
        summary["splits"][split] = {k: {"r": v[0], "mm": v[1]} for k, v in res.items()}
        if ctrl:
            summary.setdefault("controls", {})[split] = {
                k: {"r": v[0], "mm": v[1]} for k, v in ctrl.items()}

    print(f"\n=== protocol comparison  variant={args.variant} feature={args.feature} band={band}")
    print(f"{'split':<8}{'folds':>6}{'r':>9}" + "".join(f"{f'{w:g}s':>9}" for w in windows))
    for split, folds in summary["splits"].items():
        r = np.nanmean([f["r"] for f in folds.values()])
        row = "".join(f"{np.nanmean([f['mm'][f'{w:g}s'] for f in folds.values()]):9.3f}" for w in windows)
        print(f"{split:<8}{len(folds):>6}{r:9.4f}{row}")
        ctrl = summary.get("controls", {}).get(split)
        if ctrl:
            cr = float(np.nanmean([v["r"] for v in ctrl.values()]))
            crow = "".join(
                f"{np.nanmean([v['mm'][f'{w:g}s'] for v in ctrl.values()]):9.3f}" for w in windows)
            print(f"{'  ^ctrl':<8}{len(ctrl):>6}{cr:9.4f}{crow}")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1, default=float))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
