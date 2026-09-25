"""Response latency in three datasets, computed identically: neural or hardware?

MUSIN-G music peaks ~62 ms later than SparrKULee speech. That could be a real speech/music
difference or MUSIN-G's audio delivery latency. The Bach listening data (different lab, different
setup) arbitrates: a similar delay there points to the brain, none points to MUSIN-G hardware.

Same onset feature, same band, same TRF estimator everywhere. Latency is read from global field
power (std over channels), which does not depend on electrode layout. Per-listener TRFs give
confidence intervals.

    python -m eg606.eval.trf_compare --band 4,8
"""
from __future__ import annotations

import argparse
import json
import re
import logging
import sys
from pathlib import Path

import numpy as np

from eg606.eval.latency import TRF_LAGS, bach_trials, trf
from eg606.eval.linear import FS, zscore
from eg606.eval.protocol import bandpass
from eg606.eval.transfer import music_trials, speech_trials
from eg606.paths import derived_dir

log = logging.getLogger(__name__)
WINDOW_MS = (0, 400)       # search window for the peak, away from edge effects


def peak_ms(t: np.ndarray) -> float:
    lags_ms = TRF_LAGS / FS * 1000
    gfp = t.std(1)
    keep = (lags_ms >= WINDOW_MS[0]) & (lags_ms <= WINDOW_MS[1])
    return float(lags_ms[keep][np.argmax(gfp[keep])])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.eval.trf_compare")
    ap.add_argument("--band", default="4,8")
    ap.add_argument("--speech-subjects", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    band = tuple(float(x) for x in args.band.split(",")) if args.band else None

    sp_root, mu_root, ba_root = derived_dir("sparrkulee"), derived_dir("musin_g"), derived_dir("bach_silence")
    sp_feats = dict(np.load(sp_root / "audio_features.npz"))
    mu_feats = dict(np.load(mu_root / "audio_features.npz"))

    groups = {
        "speech (SparrKULee)": [list(speech_trials(p, sp_feats, "onset", band))
                                for p in sorted(sp_root.glob("sub-*.npz"))[: args.speech_subjects]],
        "music (MUSIN-G)": [[(e, y) for _, e, y in music_trials(p, mu_feats, "onset", band)]
                            for p in sorted(x for x in mu_root.glob("sub-*_bs64.npz")
                                            if re.match(r"^sub-\d+_bs64$", x.stem))],
        "music (Bach)": [list(bach_trials(p, band)) for p in sorted(ba_root.glob("sub-*.npz"))],
    }

    lags_ms = TRF_LAGS / FS * 1000
    summary = {"band": band, "lags_ms": lags_ms.tolist(), "datasets": {}}
    grand = {}
    for name, per_listener in groups.items():
        peaks = [peak_ms(trf(trials)) for trials in per_listener if trials]
        pooled = trf([t for trials in per_listener for t in trials])
        grand[name] = pooled
        rng = np.random.default_rng(0)
        boots = [np.median(rng.choice(peaks, len(peaks))) for _ in range(2000)]
        summary["datasets"][name] = {
            "listeners": len(peaks), "peaks_ms": peaks,
            "median_peak_ms": float(np.median(peaks)),
            "ci95_ms": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "pooled_peak_ms": peak_ms(pooled), "gfp": pooled.std(1).tolist()}
        log.info("%s: %d listeners, median peak %.0f ms", name, len(peaks), np.median(peaks))

    ref = "speech (SparrKULee)"
    print(f"\n=== response latency, onset feature, band={band}")
    print(f"{'dataset':<22}{'n':>4}{'median peak':>13}{'95% CI':>18}{'pooled peak':>13}")
    for name, s in summary["datasets"].items():
        print(f"{name:<22}{s['listeners']:>4}{s['median_peak_ms']:>10.0f} ms"
              f"  [{s['ci95_ms'][0]:4.0f}, {s['ci95_ms'][1]:4.0f}] ms{s['pooled_peak_ms']:>10.0f} ms")
    from scipy.stats import mannwhitneyu
    print("\nper-listener peak latency vs speech (Mann-Whitney U, two-sided):")
    for name in [n for n in summary["datasets"] if n != ref]:
        a, b = summary["datasets"][name]["peaks_ms"], summary["datasets"][ref]["peaks_ms"]
        u, pval = mannwhitneyu(a, b, alternative="two-sided")
        diff = np.median(a) - np.median(b)
        print(f"  {name:<20} median difference {diff:+5.0f} ms  U={u:.0f}  p={pval:.3g}")
        summary["datasets"][name]["vs_speech"] = {"median_diff_ms": float(diff), "U": float(u), "p": float(pval)}
    g_ref = grand[ref].std(1)
    print("\nshift that best aligns each music GFP to speech (positive = music later):")
    for name in [n for n in grand if n != ref]:
        g = grand[name].std(1)
        xc = [np.corrcoef(np.roll(g_ref, k), g)[0, 1] for k in range(-12, 13)]
        k = int(np.argmax(xc)) - 12
        print(f"  {name:<20} {k / FS * 1000:+5.0f} ms (r={max(xc):.3f})")
        summary["datasets"][name]["gfp_shift_vs_speech_ms"] = k / FS * 1000
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
