"""Is the inverted speech->music transfer a latency mismatch?

Two tests, both in the shared 64-channel montage with the onset feature.

1. Shift sweep (causal). Train the speech backward decoder, apply it to music while shifting the
   music target by delta ms. If speech and music responses share a spatial pattern but differ in
   latency, accuracy should swing from below chance to above chance at some delta.

2. Forward TRFs (descriptive). Estimate each dataset's temporal response function
   EEG(t) = sum_lag trf(lag) * stimulus(t - lag), average over listeners, and compare peak latencies
   and the lag that best aligns the two time courses.

    python -m eg606.eval.latency --band 4,8
"""
from __future__ import annotations

import argparse
import json
import re
import logging
import sys
from pathlib import Path

import mne
import numpy as np

from eg606.eval.linear import FS, accumulate, match_mismatch, predict, solve, zscore
from eg606.eval.protocol import bandpass
from eg606.eval.transfer import LAMBDAS, music_trials, openmiir_trials, speech_trials
from eg606.paths import derived_dir

log = logging.getLogger(__name__)
SHIFTS = np.arange(-20, 21, 2)                  # samples at 64 Hz: -312 .. +312 ms
TRF_LAGS = np.arange(-6, 33)                    # -94 .. +500 ms
CLUSTER = ["Fz", "FCz", "Cz", "FC1", "FC2", "C1", "C2", "F1", "F2"]


def bach_trials(path: Path, band):
    d = np.load(path, allow_pickle=True)
    for m in json.loads(str(d["meta"])):
        eeg = d[f"eeg{m['trial']:02d}"].astype(np.float64)
        y = d[f"onset{m['trial']:02d}"].astype(np.float64)
        n = min(eeg.shape[1], len(y))
        eeg, y = eeg[:, :n], y[:n]
        if band:
            eeg, y = bandpass(eeg, *band), bandpass(y[None], *band)[0]
        yield zscore(eeg), zscore(y)



def shifted(pred, y, d):
    """Pair pred(t) with y(t - d): positive d assumes the music response is d samples later."""
    n = min(len(pred), len(y))
    if d >= 0:
        return pred[d:n], y[: n - d]
    return pred[: n + d], y[-d:n]


def trf(trials, lam: float = 1e2) -> np.ndarray:
    """Forward model per listener-set: returns (lags, channels)."""
    L = len(TRF_LAGS)
    sts = ste = None
    for eeg, y in trials:
        n = min(eeg.shape[1], len(y))
        S = np.zeros((n, L))
        for i, lag in enumerate(TRF_LAGS):
            if lag >= 0:
                S[lag:, i] = y[: n - lag]
            else:
                S[: n + lag, i] = y[-lag:n]
        sts = S.T @ S if sts is None else sts + S.T @ S
        ste = S.T @ eeg[:, :n].T if ste is None else ste + S.T @ eeg[:, :n].T
    return np.linalg.solve(sts + lam * np.trace(sts) / L * np.eye(L), ste)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.eval.latency")
    ap.add_argument("--speech-subjects", type=int, default=20)
    ap.add_argument("--feature", default="onset")
    ap.add_argument("--band", default=None)
    ap.add_argument("--windows", default="10,30")
    ap.add_argument("--music", default="musin_g", choices=["musin_g", "bach", "openmiir"],
                    help="which music dataset the speech decoder is applied to")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    band = tuple(float(x) for x in args.band.split(",")) if args.band else None
    windows = [float(w) for w in args.windows.split(",")]

    sp_root, mu_root = derived_dir("sparrkulee"), derived_dir("musin_g")
    sp_feats = dict(np.load(sp_root / "audio_features.npz"))
    mu_feats = dict(np.load(mu_root / "audio_features.npz"))
    sp_subs = sorted(sp_root.glob("sub-*.npz"))[: args.speech_subjects + 4]
    train_subs, val_subs = sp_subs[: args.speech_subjects], sp_subs[args.speech_subjects:]

    # ---- speech decoder (same recipe as the transfer experiment)
    xtx = xty = None
    speech_for_trf = []
    for p in train_subs:
        for eeg, y in speech_trials(p, sp_feats, args.feature, band):
            xtx, xty = accumulate(eeg, y, xtx, xty)
            speech_for_trf.append((eeg, y))
    best, best_r = LAMBDAS[len(LAMBDAS) // 2], -np.inf
    for lam in LAMBDAS:
        w = solve(xtx, xty, lam)
        rs = [np.corrcoef(predict(e, w, len(y)), y)[0, 1]
              for p in val_subs for e, y in speech_trials(p, sp_feats, args.feature, band)]
        if np.mean(rs) > best_r:
            best, best_r = lam, float(np.mean(rs))
    w = solve(xtx, xty, best)
    log.info("speech decoder ready (lambda %.0e, held-out speech r %.4f)", best, best_r)

    # ---- music trials in the same montage
    music, owners = [], []
    if args.music == "openmiir":
        om_root = derived_dir("openmiir")
        om_feats = {v: dict(np.load(om_root / f"audio_features_{v}.npz"))
                    for v in ("v1", "v2") if (om_root / f"audio_features_{v}.npz").exists()}
        for p in sorted(om_root.glob("sub-*.npz")):
            for _, e, y in openmiir_trials(p, om_feats, args.feature, band):
                music.append((e, y))
                owners.append(p.stem)
    elif args.music == "bach":
        for p in sorted(derived_dir("bach_silence").glob("sub-*.npz")):
            for e, y in bach_trials(p, band):
                music.append((e, y))
                owners.append(p.stem)
    else:
        for p in sorted(x for x in mu_root.glob("sub-*_bs64.npz") if re.match(r"^sub-\d+_bs64$", x.stem)):
            for _, e, y in music_trials(p, mu_feats, args.feature, band):
                music.append((e, y))
                owners.append(p.stem.split("_")[0])
    preds = [(predict(e, w, len(y)), y) for e, y in music]
    log.info("%d music trials scored", len(music))

    # ---- 1. shift sweep
    sweep = []
    for d in SHIFTS:
        rs, mm = [], {f"{x:g}s": [0, 0] for x in windows}
        for pred, y in preds:
            p, yy = shifted(pred, y, int(d))
            if len(yy) < 64 or p.std() == 0:
                continue
            rs.append(np.corrcoef(p, yy)[0, 1])
            for x in windows:
                h, t = match_mismatch(p, yy, int(x * FS))
                mm[f"{x:g}s"][0] += h
                mm[f"{x:g}s"][1] += t
        row = {"shift_ms": float(d / FS * 1000), "r": float(np.mean(rs)),
               **{k: v[0] / v[1] for k, v in mm.items() if v[1]}}
        sweep.append(row)
        log.info("shift %+5.0f ms  r=%+.4f  %s", row["shift_ms"], row["r"],
                 {k: round(row[k], 3) for k in mm if k in row})

    # ---- 1b. split-half: the shift is chosen on listeners it is NOT scored on
    subs = sorted(set(owners))
    halves = [set(subs[: len(subs) // 2]), set(subs[len(subs) // 2:])]
    key = f"{windows[-1]:g}s"

    def acc_at(d, keep):
        hits = total = 0
        for (pred, y), who in zip(preds, owners):
            if who not in keep:
                continue
            pp, yy = shifted(pred, y, int(d))
            if len(yy) < 64 or pp.std() == 0:
                continue
            h, tt = match_mismatch(pp, yy, int(windows[-1] * FS))
            hits += h
            total += tt
        return hits / total if total else float("nan")

    split_half = []
    for a, b in ((0, 1), (1, 0)):
        d_best = max(SHIFTS, key=lambda d: acc_at(d, halves[a]))
        split_half.append({"select_on": a, "shift_ms": float(d_best / FS * 1000),
                           "test_unshifted": acc_at(0, halves[b]),
                           "test_at_selected_shift": acc_at(d_best, halves[b])})
        log.info("split-half: shift %+.0f ms chosen on half %d -> half %d: %.3f (unshifted %.3f)",
                 d_best / FS * 1000, a, b, split_half[-1]["test_at_selected_shift"],
                 split_half[-1]["test_unshifted"])

    # ---- 2. forward TRFs
    trf_speech = trf(speech_for_trf)
    trf_music = trf(music)
    names = mne.channels.make_standard_montage("biosemi64").ch_names
    idx = [names.index(c) for c in CLUSTER]
    lags_ms = TRF_LAGS / FS * 1000
    ts, tm = trf_speech[:, idx].mean(1), trf_music[:, idx].mean(1)
    gfp_s, gfp_m = trf_speech.std(1), trf_music.std(1)
    peak_s, peak_m = lags_ms[np.argmax(gfp_s)], lags_ms[np.argmax(gfp_m)]
    xc = [float(np.corrcoef(np.roll(ts, k), tm)[0, 1]) for k in range(-10, 11)]
    best_k = int(np.argmax(xc)) - 10
    # A GFP peak marks the largest deflection without saying which way it points, so two datasets
    # can be compared at peaks of opposite polarity and look anticorrelated when the pattern is the
    # same. Orient both to negative over the fronto-central cluster, as eval/topo.py does.
    def _orient(topo):
        return -topo if topo[idx].mean() > 0 else topo

    spatial = float(np.corrcoef(_orient(trf_speech[np.argmax(gfp_s)]),
                                _orient(trf_music[np.argmax(gfp_m)]))[0, 1])

    print(f"\n=== latency analysis, music={args.music}, feature={args.feature}, band={band}")
    print("shift sweep (speech decoder on music; positive = music response later):")
    print(f"{'shift ms':>9}{'r':>9}" + "".join(f"{f'{w:g}s':>8}" for w in windows))
    for row in sweep:
        print(f"{row['shift_ms']:9.0f}{row['r']:9.4f}" + "".join(f"{row.get(f'{w:g}s', float('nan')):8.3f}" for w in windows))
    bestrow = max(sweep, key=lambda r: r.get(f"{windows[-1]:g}s", 0))
    print(f"\nbest shift: {bestrow['shift_ms']:+.0f} ms -> {bestrow.get(f'{windows[-1]:g}s'):.3f} at {windows[-1]:g}s "
          f"(unshifted: {[r for r in sweep if r['shift_ms'] == 0][0].get(f'{windows[-1]:g}s'):.3f})")
    for s in split_half:
        print(f"split-half: shift {s['shift_ms']:+.0f} ms chosen on half {s['select_on']}; other half "
              f"{s['test_unshifted']:.3f} -> {s['test_at_selected_shift']:.3f} at {key}")
    print(f"TRF global-field-power peak: speech {peak_s:.0f} ms, music {peak_m:.0f} ms "
          f"(difference {peak_m - peak_s:+.0f} ms)")
    print(f"fronto-central TRF best alignment: music vs speech shifted {best_k / FS * 1000:+.0f} ms "
          f"(r={max(xc):.3f}; unshifted r={xc[10]:.3f})")
    print(f"spatial pattern similarity at the two peaks: r={spatial:.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "band": band, "feature": args.feature, "sweep": sweep,
            "lags_ms": lags_ms.tolist(), "trf_speech": trf_speech.tolist(), "trf_music": trf_music.tolist(),
            "peak_ms": {"speech": float(peak_s), "music": float(peak_m)},
            "xcorr": xc, "best_align_ms": best_k / FS * 1000, "spatial_r": spatial,
            "split_half": split_half}, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
