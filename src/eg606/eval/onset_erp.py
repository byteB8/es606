"""Is MUSIN-G's audio really late? Ask the evoked response, not the decoder.

The +62 ms offset was found by sweeping a speech decoder over music and watching accuracy move.
A reviewer can call that a modelling artefact, so this measures the same thing with an analysis
that shares no machinery with it: the response evoked by the *first sound of a song*.

The auditory N1/P2 complex is one of the most replicated latencies in EEG (N1 near 100 ms, P2 near
180 ms at the vertex). If MUSIN-G's onset response sits ~62 ms later than that, and later than the
same measurement in an independent dataset recorded on the same cap family (NMED-T, Stanford),
the delay is in the recording, not in the model.

Two confounds are handled here:
  * a song that fades in has no sharp onset, which would delay N1 for acoustic reasons. Latency is
    therefore reported twice: relative to the trigger, and relative to the moment the audio
    actually rises (the acoustic onset). Only the trigger-relative number can indict the hardware.
  * electrode layout. Latency is read from global field power, which does not depend on it.

    python -m eg606.eval.onset_erp --out results/onset_erp.json --fig results/onset_erp.png
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import mne
import numpy as np

from eg606.audio.features import read_wav
from eg606.paths import derived_dir, raw_dir
from eg606.prep.musin_g import RIM, song_durations, trials as musin_g_trials

log = logging.getLogger(__name__)

FS_ERP = 250.0
TMIN, TMAX = -0.3, 0.7
BASELINE = (-0.3, -0.02)
ERP_BAND = (0.5, 20.0)
VERTEX = ["E4", "E5", "E6", "E7", "E11", "E12", "E13", "E106", "E112"]
N1_WIN = (0.050, 0.230)   # search windows, generous enough to hold a delayed response
P2_WIN = (0.120, 0.360)
GFP_WIN = (0.040, 0.300)
PAD_S = 10.0              # s read either side of an onset, so the 0.5 Hz filter has room
RISE_FRACTION = 0.25      # of the first second's RMS: when the audio has actually begun


def epochs_from_raw(raw: mne.io.BaseRaw, onsets_s: list[float]):
    """((trials, channels, time) around each onset, baseline corrected; channel names)."""
    if "E129" in raw.ch_names:
        raw.rename_channels({"E129": "Cz"})
    raw.set_montage("GSN-HydroCel-129", on_missing="ignore", verbose="ERROR")
    raw.drop_channels([c for c in RIM + ["Cz"] if c in raw.ch_names])
    raw.filter(*ERP_BAND, fir_design="firwin", verbose="ERROR")
    raw.set_eeg_reference("average", verbose="ERROR")
    raw.resample(FS_ERP, npad="auto", verbose="ERROR")

    a, b = int(round(TMIN * FS_ERP)), int(round(TMAX * FS_ERP))
    ba, bb = int(round((BASELINE[0] - TMIN) * FS_ERP)), int(round((BASELINE[1] - TMIN) * FS_ERP))
    out = []
    for t in onsets_s:
        s = int(round(t * FS_ERP))
        if s + a < 0 or s + b > raw.n_times:
            continue
        e = raw.get_data(start=s + a, stop=s + b)
        out.append(e - e[:, ba:bb].mean(1, keepdims=True))
    ep = np.stack(out) if out else np.zeros((0, len(raw.ch_names), b - a))
    return ep, raw.ch_names


def acoustic_onset(x: np.ndarray, sr: int) -> float:
    """Seconds from the file's start until the audio has genuinely begun."""
    win = max(1, int(0.005 * sr))
    n = len(x) // win * win
    rms = np.sqrt((x[:n].reshape(-1, win) ** 2).mean(1))
    ref = np.sqrt((x[: sr].astype(np.float64) ** 2).mean())
    above = np.where(rms >= RISE_FRACTION * ref)[0]
    return float(above[0] * win / sr) if len(above) else 0.0


def musin_g_epochs(subjects: int | None, max_rise: float | None = None):
    root = raw_dir("musin_g")
    durations = song_durations(root)
    rise = {}
    for i in range(1, 13):
        x, sr = read_wav(root / "Code" / "ESongs" / f"{i}.esh.wav")
        rise[i] = acoustic_onset(x, sr)
    sharp = sorted(k for k, v in rise.items() if max_rise is None or v <= max_rise)
    if max_rise is not None:
        log.info("songs starting within %.0f ms: %s", 1000 * max_rise, sharp)
    subs = sorted(Path(p).name for p in glob.glob(str(root / "sourcedata" / "sub-*")))[:subjects]
    per_subject, per_subject_ac, ch_names = {}, {}, []
    for sub in subs:
        setf = glob.glob(str(root / "sourcedata" / sub / "eeg" / "*_eeg.set"))[0]
        tr = [t for t in musin_g_trials(root, sub, durations) if t["song"] in sharp]
        if not tr:
            continue
        raw = mne.io.read_raw_eeglab(setf, preload=True, verbose="ERROR")
        ep, names = epochs_from_raw(raw.copy(), [t["onset"] for t in tr])
        ep_ac, _ = epochs_from_raw(raw, [t["onset"] + rise[t["song"]] for t in tr])
        if len(ep):
            per_subject[sub] = ep.mean(0)
            per_subject_ac[sub] = ep_ac.mean(0)
            ch_names = names
        log.info("%s: %d onset epochs", sub, len(ep))
    log.info("MUSIN-G acoustic rise times: median %.0f ms, max %.0f ms",
             1000 * np.median(list(rise.values())), 1000 * max(rise.values()))
    return per_subject, per_subject_ac, rise, ch_names


def nmed_t_epochs(subjects: int | None):
    """NMED-T's raw sessions: DI21..DI30 mark the audio onset of that song."""
    import h5py
    from eg606.prep.nmed import _raw_events

    root = raw_dir("nmed_t")
    per_subject: dict[str, list[np.ndarray]] = {}
    ch_names: list[str] = []
    files = sorted(root.glob("*_*_raw.mat"))
    seen: set[str] = set()
    for path in files:
        sub = f"sub-S{int(path.name.split('_')[0]):02d}"
        if subjects is not None and sub not in seen and len(seen) >= subjects:
            continue
        seen.add(sub)
        ev, fs, shape = _raw_events(path)
        onsets = [s for c, s in ev if c.startswith("DI") and c[2:].isdigit() and 21 <= int(c[2:]) <= 30]
        if not onsets:
            continue
        pad = int(PAD_S * fs)
        with h5py.File(path, "r") as f:
            X = f["X"]
            for s in onsets:
                a, b = max(0, s - pad), min(shape[0], s + pad)
                seg = np.asarray(X[a:b, :]).T.astype(np.float64)
                info = mne.create_info([f"E{i + 1}" for i in range(seg.shape[0])], fs, "eeg")
                raw = mne.io.RawArray(seg, info, verbose="ERROR")
                ep, names = epochs_from_raw(raw, [(s - a) / fs])
                if len(ep):
                    per_subject.setdefault(sub, []).append(ep[0])
                    ch_names[:] = names
        log.info("%s: %d onset epochs so far", sub, len(per_subject.get(sub, [])))
    return {k: np.stack(v).mean(0) for k, v in per_subject.items() if v}, ch_names


def refine(y: np.ndarray, i: int) -> float:
    """Sub-sample peak position by fitting a parabola to the three samples around index i."""
    if 0 < i < len(y) - 1:
        d = y[i - 1] - 2 * y[i] + y[i + 1]
        if d != 0:
            return i + 0.5 * (y[i - 1] - y[i + 1]) / d
    return float(i)


def peaks(avg: np.ndarray, ch_names: list[str], times: np.ndarray) -> dict:
    gfp = avg.std(0)
    keep = (times >= GFP_WIN[0]) & (times <= GFP_WIN[1])
    i = int(np.argmax(gfp[keep])) + int(np.argmax(keep))
    out = {"gfp_peak_ms": 1000 * np.interp(refine(gfp, i), np.arange(len(times)), times)}
    idx = [ch_names.index(c) for c in VERTEX if c in ch_names]
    if idx:
        v = avg[idx].mean(0)
        for name, win, sign in (("n1", N1_WIN, -1), ("p2", P2_WIN, +1)):
            k = (times >= win[0]) & (times <= win[1])
            j = int(np.argmax(sign * v[k])) + int(np.argmax(k))
            out[f"{name}_ms"] = 1000 * np.interp(refine(sign * v, j), np.arange(len(times)), times)
            out[f"{name}_uv"] = float(v[j] * 1e6)
        out["vertex"] = v.tolist()
    out["gfp"] = gfp.tolist()
    return out


def summarise(per_subject: dict, ch_names: list[str], times: np.ndarray, label: str) -> dict:
    subs = sorted(per_subject)
    grand = np.mean([per_subject[s] for s in subs], axis=0)
    g = peaks(grand, ch_names, times)
    rng = np.random.default_rng(0)
    boot = {k: [] for k in ("gfp_peak_ms", "n1_ms", "p2_ms") if k in g}
    for _ in range(1000):
        pick = rng.choice(subs, len(subs))
        p = peaks(np.mean([per_subject[s] for s in pick], axis=0), ch_names, times)
        for k in boot:
            boot[k].append(p[k])
    out = {"n_listeners": len(subs), "grand": {k: v for k, v in g.items() if not isinstance(v, list)},
           "ci95": {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                    for k, v in boot.items()},
           "per_listener": {s: {k: v for k, v in peaks(per_subject[s], ch_names, times).items()
                                if not isinstance(v, list)} for s in subs},
           "gfp": g["gfp"], "vertex": g.get("vertex")}
    log.info("%-18s n=%2d  GFP peak %6.1f ms  N1 %6.1f ms  P2 %6.1f ms", label, len(subs),
             g["gfp_peak_ms"], g.get("n1_ms", float("nan")), g.get("p2_ms", float("nan")))
    return out


def figure(res: dict, times: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    colors = {"musin_g": "#c0392b", "musin_g_acoustic": "#e67e22", "nmed_t": "#2c6fbb"}
    labels = {"musin_g": "MUSIN-G (trigger)", "musin_g_acoustic": "MUSIN-G (acoustic onset)",
              "nmed_t": "NMED-T (trigger)"}
    for key, r in res.items():
        if not isinstance(r, dict) or "gfp" not in r:
            continue
        axes[0].plot(times * 1000, np.array(r["gfp"]) * 1e6, color=colors.get(key), label=labels.get(key, key))
        if r.get("vertex"):
            axes[1].plot(times * 1000, np.array(r["vertex"]) * 1e6, color=colors.get(key))
        axes[0].axvline(r["grand"]["gfp_peak_ms"], color=colors.get(key), ls=":", lw=1)
    for ax in axes:
        ax.axvline(0, color="k", lw=0.8)
        ax.axvspan(90, 110, color="0.85", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("global field power (uV)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Response to the first sound of a song (grey band: canonical N1, 90-110 ms)")
    axes[1].set_ylabel("vertex cluster (uV)")
    axes[1].set_xlabel("time from marked onset (ms)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    log.info("wrote %s", path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.eval.onset_erp")
    ap.add_argument("--datasets", default="musin_g,nmed_t")
    ap.add_argument("--subjects", type=int, default=None, help="limit, for a quick check")
    ap.add_argument("--max-rise-ms", type=float, default=None,
                    help="keep only MUSIN-G songs whose audio reaches full level within this time")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fig", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    times = np.arange(int(round(TMIN * FS_ERP)), int(round(TMAX * FS_ERP))) / FS_ERP
    want = args.datasets.split(",")
    res: dict = {"fs": FS_ERP, "times_s": times.tolist()}
    ch_names = None

    if "musin_g" in want:
        trig, acoustic, rise, ch_names = musin_g_epochs(
            args.subjects, None if args.max_rise_ms is None else args.max_rise_ms / 1000)
        res["musin_g"] = summarise(trig, ch_names, times, "MUSIN-G trigger")
        res["musin_g_acoustic"] = summarise(acoustic, ch_names, times, "MUSIN-G acoustic")
        res["musin_g_rise_ms"] = {str(k): 1000 * v for k, v in rise.items()}
    if "nmed_t" in want:
        nm, nm_names = nmed_t_epochs(args.subjects)
        if nm:
            res["nmed_t"] = summarise(nm, nm_names, times, "NMED-T trigger")

    print("\n=== onset-evoked response, grand average over listeners")
    print(f"{'dataset':<26}{'n':>4}{'GFP peak':>12}{'95% CI':>18}{'N1':>10}{'P2':>10}")
    for key in ("nmed_t", "musin_g", "musin_g_acoustic"):
        r = res.get(key)
        if not r:
            continue
        ci = r["ci95"]["gfp_peak_ms"]
        print(f"{key:<26}{r['n_listeners']:>4}{r['grand']['gfp_peak_ms']:>9.1f} ms"
              f"  [{ci[0]:5.1f},{ci[1]:6.1f}]{r['grand'].get('n1_ms', float('nan')):>7.1f} ms"
              f"{r['grand'].get('p2_ms', float('nan')):>7.1f} ms")
    if "musin_g" in res and "nmed_t" in res:
        d = res["musin_g"]["grand"]["gfp_peak_ms"] - res["nmed_t"]["grand"]["gfp_peak_ms"]
        da = res["musin_g_acoustic"]["grand"]["gfp_peak_ms"] - res["nmed_t"]["grand"]["gfp_peak_ms"]
        g_m, g_n = np.array(res["musin_g"]["gfp"]), np.array(res["nmed_t"]["gfp"])
        xc = [float(np.corrcoef(np.roll(g_n, k), g_m)[0, 1]) for k in range(-40, 41)]
        lag = (int(np.argmax(xc)) - 40) / FS_ERP * 1000
        res["musin_g_minus_nmed_t_ms"] = {"gfp_peak": d, "gfp_peak_acoustic": da, "xcorr_lag": lag}
        print(f"\nMUSIN-G later than NMED-T by {d:+.1f} ms at the GFP peak "
              f"({da:+.1f} ms once MUSIN-G's own audio rise time is removed)")
        print(f"best alignment of the two GFP time courses: {lag:+.1f} ms (r={max(xc):.3f})")
        from scipy.stats import mannwhitneyu
        a = [v["gfp_peak_ms"] for v in res["musin_g"]["per_listener"].values()]
        b = [v["gfp_peak_ms"] for v in res["nmed_t"]["per_listener"].values()]
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        res["per_listener_test"] = {"U": float(u), "p": float(p),
                                    "median_diff_ms": float(np.median(a) - np.median(b))}
        print(f"per-listener GFP peak, MUSIN-G vs NMED-T: median difference "
              f"{np.median(a) - np.median(b):+.1f} ms, U={u:.0f}, p={p:.3g}")
    if args.fig:
        Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
        figure(res, times, Path(args.fig))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
