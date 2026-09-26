"""Where on the scalp does the transfer happen, and what does the decoder use?

The latency analysis says speech and music responses differ in *when*. This asks *where*, which is
the part a reviewer needs to believe something neural transferred rather than something numerical:

  1. forward TRF topographies at each dataset's own peak. If speech and music share a
     fronto-central, auditory-looking pattern, the decoder is reading the same generator.
  2. the backward decoder's own weights, summed over lags per channel. This is what the model
     actually leans on, and it is not the same object as the TRF.
  3. how similar those patterns are across datasets, with a listener-level bootstrap, so
     "the patterns look alike" becomes a number with an interval.

Everything runs in the shared 64-channel montage, so the three datasets are directly comparable.

    python -m eg606.eval.topo --band 4,8 --out results/e7/topo.json --fig results/e7/topo.png
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import mne
import numpy as np

from eg606.eval.latency import TRF_LAGS, bach_trials, trf
from eg606.eval.linear import FS, LAGS, accumulate, predict, solve
from eg606.eval.transfer import LAMBDAS, music_trials, speech_trials
from eg606.paths import derived_dir

log = logging.getLogger(__name__)
MONTAGE = "biosemi64"
PEAK_WINDOW = (0, 400)     # ms
# A GFP peak marks the moment of largest deflection without saying which way it points, so two
# datasets can be compared at peaks of opposite polarity and look anticorrelated when the pattern
# is the same. Maps are therefore oriented to a common convention: negative over fronto-central
# electrodes, which is how an auditory N1 presents.
POLARITY_REF = ["Fz", "FCz", "Cz", "FC1", "FC2", "C1", "C2"]


def peak_index(t: np.ndarray) -> int:
    lags_ms = TRF_LAGS / FS * 1000
    keep = (lags_ms >= PEAK_WINDOW[0]) & (lags_ms <= PEAK_WINDOW[1])
    return int(np.argmax(t.std(1)[keep])) + int(np.argmax(keep))


def decoder_weights(trial_sets: list, val_sets: list) -> tuple[np.ndarray, float]:
    """Backward model over pooled listeners; |weight| summed over lags gives a channel map."""
    xtx = xty = None
    for trials in trial_sets:
        for eeg, y in trials:
            xtx, xty = accumulate(eeg, y, xtx, xty)
    best, best_r = LAMBDAS[len(LAMBDAS) // 2], -np.inf
    for lam in LAMBDAS:
        w = solve(xtx, xty, lam)
        rs = [np.corrcoef(predict(e, w, len(y)), y)[0, 1] for trials in val_sets for e, y in trials]
        if rs and np.mean(rs) > best_r:
            best, best_r = lam, float(np.mean(rs))
    w = solve(xtx, xty, best)
    n_ch = len(w) // len(LAGS)
    return np.abs(w.reshape(len(LAGS), n_ch)).sum(0), best_r


def orient(topo: np.ndarray, ch_names: list[str]) -> np.ndarray:
    idx = [ch_names.index(c) for c in POLARITY_REF if c in ch_names]
    return -topo if idx and topo[idx].mean() > 0 else topo


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def figure(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    info = mne.create_info(res["ch_names"], FS, "eeg")
    info.set_montage(MONTAGE)
    names = [n for n in res["topographies"]]
    fig, axes = plt.subplots(2, len(names), figsize=(3 * len(names), 6))
    axes = np.atleast_2d(axes)
    for j, name in enumerate(names):
        for i, (key, title) in enumerate((("trf", "TRF at peak"), ("decoder", "decoder weight"))):
            v = np.asarray(res["topographies"][name][key], dtype=float)
            mne.viz.plot_topomap(v, info, axes=axes[i, j], show=False, cmap="RdBu_r",
                                 contours=4, sensors=False)
            if i == 0:
                axes[i, j].set_title(f"{name}\n{res['topographies'][name]['peak_ms']:.0f} ms",
                                     fontsize=9)
            if j == 0:
                axes[i, j].text(-0.25, 0.5, title, transform=axes[i, j].transAxes,
                                rotation=90, va="center", fontsize=9)
    fig.suptitle("Forward response and backward decoder, same montage, three datasets", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    log.info("wrote %s", path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.eval.topo")
    ap.add_argument("--band", default="4,8")
    ap.add_argument("--speech-subjects", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fig", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    band = tuple(float(x) for x in args.band.split(",")) if args.band else None

    sp_root, mu_root = derived_dir("sparrkulee"), derived_dir("musin_g")
    ba_root = derived_dir("bach_silence")
    sp_feats = dict(np.load(sp_root / "audio_features.npz"))
    mu_feats = dict(np.load(mu_root / "audio_features.npz"))
    sp_subs = sorted(sp_root.glob("sub-*.npz"))[: args.speech_subjects + 4]

    groups = {
        "speech (SparrKULee)": [list(speech_trials(p, sp_feats, "onset", band))
                                for p in sp_subs[: args.speech_subjects]],
        "music (MUSIN-G)": [[(e, y) for _, e, y in music_trials(p, mu_feats, "onset", band)]
                            for p in sorted(x for x in mu_root.glob("sub-*_bs64.npz")
                                            if re.match(r"^sub-\d+_bs64$", x.stem))],
        "music (Bach)": [list(bach_trials(p, band)) for p in sorted(ba_root.glob("sub-*.npz"))],
    }
    speech_val = [list(speech_trials(p, sp_feats, "onset", band)) for p in sp_subs[args.speech_subjects:]]
    ch_names = mne.channels.make_standard_montage(MONTAGE).ch_names

    res = {"band": band, "ch_names": ch_names, "topographies": {}, "similarity": {}}
    per_listener_trfs = {}
    for name, per_listener in groups.items():
        trfs = [trf(t) for t in per_listener if t]
        per_listener_trfs[name] = trfs
        pooled = trf([t for trials in per_listener for t in trials])
        i = peak_index(pooled)
        # a decoder needs held-out data to pick its ridge penalty; within a dataset, use two listeners
        val = speech_val if name.startswith("speech") else per_listener[-2:]
        train = per_listener if name.startswith("speech") else per_listener[:-2]
        wmap, val_r = decoder_weights(train, val)
        res["topographies"][name] = {
            "peak_ms": float(TRF_LAGS[i] / FS * 1000), "peak_index": i,
            "trf": orient(pooled[i], ch_names).tolist(),
            "decoder": wmap.tolist(), "val_r": val_r,
            "n_listeners": len(trfs)}
        log.info("%-22s peak %5.1f ms, decoder validation r %.4f", name,
                 TRF_LAGS[i] / FS * 1000, val_r)

    ref = "speech (SparrKULee)"
    rng = np.random.default_rng(0)
    for name in groups:
        if name == ref:
            continue
        a = np.asarray(res["topographies"][ref]["trf"])
        b = np.asarray(res["topographies"][name]["trf"])
        ia, ib = res["topographies"][ref]["peak_index"], res["topographies"][name]["peak_index"]
        ta, tb = per_listener_trfs[ref], per_listener_trfs[name]
        boots = []
        for _ in range(2000):
            # the peak index is the pooled one, held fixed: re-picking it per resample lets the
            # bootstrap wander onto a deflection of the other polarity and the cosine flips sign
            ma = np.mean([ta[i] for i in rng.integers(0, len(ta), len(ta))], axis=0)
            mb = np.mean([tb[i] for i in rng.integers(0, len(tb), len(tb))], axis=0)
            boots.append(cosine(orient(ma[ia], ch_names), orient(mb[ib], ch_names)))
        res["similarity"][name] = {
            "trf_cosine_vs_speech": cosine(a, b),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "decoder_cosine_vs_speech": cosine(
                np.asarray(res["topographies"][ref]["decoder"]),
                np.asarray(res["topographies"][name]["decoder"]))}
        log.info("%-22s TRF cosine vs speech %+0.3f", name, res["similarity"][name]["trf_cosine_vs_speech"])

    print(f"\n=== scalp patterns, band={band}, montage={MONTAGE}")
    print(f"{'dataset':<24}{'n':>4}{'TRF peak':>11}{'decoder val r':>15}")
    for name, r in res["topographies"].items():
        print(f"{name:<24}{r['n_listeners']:>4}{r['peak_ms']:>8.0f} ms{r['val_r']:>15.4f}")
    print("\nspatial similarity to speech (cosine of the mean-removed maps):")
    for name, r in res["similarity"].items():
        print(f"  {name:<22} TRF {r['trf_cosine_vs_speech']:+.3f} "
              f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]   "
              f"decoder {r['decoder_cosine_vs_speech']:+.3f}")
    if args.fig:
        Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
        figure(res, Path(args.fig))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
