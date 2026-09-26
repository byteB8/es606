"""Preprocess NMED-T and NMED-H into the epoch format the leakage audit reads.

Facts established by inspection (tools/probe_nmed*.py):
  * NMED-T `songNN_Imputed.mat` is MATLAB v5 and holds `dataNN` (125 channels, time, 20 listeners)
    at 125 Hz with `subsNN` naming the listeners; songs are numbered 21-30.
  * NMED-H ships the same layout per stimulus *and per repetition* -- `dataNN_a`, `dataNN_b`,
    12 listeners each -- in one zip per stimulus version (orig, rev, meas, phase). A listener heard
    only one version of a given song, so the four zips together give each listener four distinct
    stimuli, each heard twice in separate blocks. Two hearings of one stimulus are the control
    MUSIN-G cannot provide: the sound is identical and only the recording block differs.
  * NMED-T's raw sessions are MATLAB v7.3 and carry EGI triggers in `DIN_1`: `DI11` opens a trial,
    the following `DI21`..`DI30` marks the audio onset of that song, and `DIN1` closes it. The gap
    before an audio onset is silence, which is what the leakage control needs.

    python -m eg606.prep.nmed t            # clean NMED-T
    python -m eg606.prep.nmed h            # clean NMED-H, "orig" condition
    python -m eg606.prep.nmed t-raw        # pre-song silence from the raw sessions
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import re
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

from eg606.paths import derived_dir, raw_dir

log = logging.getLogger(__name__)

FS_OUT = 64.0            # the rate every other dataset here is held at
T_SONGS = range(21, 31)  # NMED-T stimulus numbering
H_REPS = ("a", "b")
# NMED-H numbers every version separately -- orig 21-24, rev 25-28, phase 29-32, meas 33-36 -- so
# the base song is recovered by position within the block of four rather than by the raw number.
H_VERSIONS = ("orig", "rev", "meas", "phase")
H_MEMBER = re.compile(r"song(\d+)_([ab])_Imputed\.mat$")
MAX_SILENCE = 10.0       # s before an onset to keep, matching MUSIN-G's PRE_SILENCE
MIN_SILENCE = 3.0        # s below which a silence window is not worth keeping
BAND = (0.5, 32.0)       # raw sessions only; the Imputed releases are already filtered


def to_fs_out(x: np.ndarray, fs_in: float) -> np.ndarray:
    """(channels, time) resampled to FS_OUT. 125 -> 64 Hz is not a neat ratio; polyphase handles it."""
    if fs_in == FS_OUT:
        return x
    g = np.gcd(int(fs_in), int(FS_OUT))
    return resample_poly(x, int(FS_OUT) // g, int(fs_in) // g, axis=-1)


def write(dest: Path, trials: list[dict]) -> None:
    data = {t["key"]: t.pop("eeg") for t in trials}
    np.savez_compressed(dest, fs=FS_OUT, meta=json.dumps(trials), **data)


def collect(per_subject: dict[str, list[dict]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for sub, trials in sorted(per_subject.items()):
        trials.sort(key=lambda t: (t["song"], t.get("version") or "", t.get("rep") or ""))
        for i, t in enumerate(trials):
            t["key"] = f"x{i:03d}"
        write(out / f"{sub}.npz", trials)
        log.info("%s -> %d trials", sub, len(trials))


def cmd_t(args) -> int:
    """NMED-T: one Imputed file per song, all listeners stacked on the last axis."""
    root, out = raw_dir("nmed_t"), derived_dir("nmed_t")
    per_subject: dict[str, list[dict]] = {}
    for song in T_SONGS:
        m = sio.loadmat(str(root / f"song{song}_Imputed.mat"), squeeze_me=True)
        x, fs = m[f"data{song}"], float(np.asarray(m["fs"]).ravel()[0])
        subs = [str(s) for s in np.asarray(m[f"subs{song}"]).ravel()]
        log.info("song %d: %s at %g Hz, %d listeners", song, x.shape, fs, len(subs))
        for i, sub in enumerate(subs):
            eeg = to_fs_out(np.ascontiguousarray(x[:, :, i], dtype=np.float64), fs)
            per_subject.setdefault(f"sub-{sub}", []).append(
                {"song": song, "rep": None, "eeg": eeg.astype(np.float32)})
        del m, x
    collect(per_subject, out)
    return 0


def cmd_h(args) -> int:
    """NMED-H: every stimulus version, so each listener contributes four stimuli heard twice."""
    root, out = raw_dir("nmed_h"), derived_dir("nmed_h")
    per_subject: dict[str, list[dict]] = {}
    heard: dict[str, set] = {}
    for version in (args.versions or H_VERSIONS):
        zpath = root / f"CleanEEG_aggregatedByStimulus_{version}.zip"
        with zipfile.ZipFile(zpath) as z:
            members = sorted(i.filename for i in z.infolist()
                             if H_MEMBER.search(i.filename) and not i.filename.startswith("__"))
            numbers = sorted({int(H_MEMBER.search(n).group(1)) for n in members})
            for name in members:
                num, rep = H_MEMBER.search(name).groups()
                num = int(num)
                song = numbers.index(num) + 1           # 1..4, the underlying piece of music
                m = sio.loadmat(io.BytesIO(z.read(name)), squeeze_me=True)
                x, fs = m[f"data{num}_{rep}"], float(np.asarray(m["fs"]).ravel()[0])
                subs = [str(v) for v in np.asarray(m[f"subs{num}_{rep}"]).ravel()]
                log.info("%s stim %d (song %d) rep %s: %s at %g Hz, %d listeners",
                         version, num, song, rep, x.shape, fs, len(subs))
                for i, sub in enumerate(subs):
                    eeg = to_fs_out(np.ascontiguousarray(x[:, :, i], dtype=np.float64), fs)
                    per_subject.setdefault(f"sub-{sub}", []).append(
                        {"song": song, "version": version, "rep": rep,
                         "stimulus": f"{version}{num}", "eeg": eeg.astype(np.float32)})
                    heard.setdefault(sub, set()).add(f"{version}{num}")
                del m, x
    n = [len(v) for v in heard.values()]
    log.info("stimuli per listener: min %d, median %d, max %d (%d listeners)",
             min(n), int(np.median(n)), max(n), len(heard))
    collect(per_subject, out)
    return 0


def _raw_events(path: Path):
    """[(code, sample)] from the EGI DIN_1 trigger table of a v7.3 session file."""
    import h5py
    with h5py.File(path, "r") as f:
        din = f["DIN_1"]
        ev = []
        for c in range(din.shape[0]):
            code = "".join(chr(v) for v in np.asarray(f[din[c, 0]]).ravel())
            ev.append((code, int(np.asarray(f[din[c, 1]]).ravel()[0])))
        return sorted(ev, key=lambda x: x[1]), float(np.asarray(f["fs"]).ravel()[0]), f["X"].shape


# DI11 opens a trial ~1.2 s before the audio and D128 is a 60 s clock tick: neither ends the
# silence. What does is the listener's rating keypress (DIN5..DIN9) closing the previous trial.
IGNORE_CODES = {"DI11", "D128"}


def silence_spans(ev, fs, n_samples):
    """(song, start, stop) sample spans of the silence preceding each audio onset."""
    onsets = [(int(c[2:]), s) for c, s in ev if c.startswith("DI") and c[2:].isdigit()
              and 21 <= int(c[2:]) <= 30]
    spans = []
    for song, s in onsets:
        earlier = [t for c, t in ev if t < s and c not in IGNORE_CODES and t != s]
        floor = max(earlier) if earlier else 0
        start = max(floor, s - int(MAX_SILENCE * fs))
        if (s - start) / fs >= MIN_SILENCE:
            spans.append((song, start, s))
    return spans


def cmd_t_raw(args) -> int:
    """Pre-song silence from NMED-T's raw sessions: the control that needs the untrimmed recording."""
    import mne
    from eg606.prep.musin_g import RIM

    root, out = raw_dir("nmed_t"), derived_dir("nmed_t")
    per_subject: dict[str, list[dict]] = {}
    files = sorted(root.glob("*_*_raw.mat"))
    for path in files:
        import h5py
        ev, fs, shape = _raw_events(path)
        spans = silence_spans(ev, fs, shape[0])
        sub = f"sub-S{int(path.name.split('_')[0]):02d}"
        if not spans:
            log.warning("%s: no usable silence spans", path.name)
            continue
        with h5py.File(path, "r") as f:
            X = f["X"]
            for song, a, b in spans:
                seg = np.asarray(X[a:b, :]).T.astype(np.float64)   # (channels, time)
                info = mne.create_info([f"E{i + 1}" for i in range(seg.shape[0])], fs, "eeg")
                raw = mne.io.RawArray(seg, info, verbose="ERROR")
                if "E129" in raw.ch_names:
                    raw.rename_channels({"E129": "Cz"})
                raw.set_montage("GSN-HydroCel-129", on_missing="ignore", verbose="ERROR")
                raw.drop_channels([c for c in RIM + ["Cz"] if c in raw.ch_names])
                raw.filter(*BAND, fir_design="firwin", verbose="ERROR")
                raw.set_eeg_reference("average", verbose="ERROR")
                raw.resample(FS_OUT, npad="auto", verbose="ERROR")
                per_subject.setdefault(sub, []).append(
                    {"song": song, "rep": None, "silence": True,
                     "seconds": (b - a) / fs, "eeg": raw.get_data().astype(np.float32)})
        log.info("%s: %d silence spans, %.1f-%.1f s", path.name, len(spans),
                 min((b - a) / fs for _, a, b in spans), max((b - a) / fs for _, a, b in spans))

    out.mkdir(parents=True, exist_ok=True)
    for sub, trials in sorted(per_subject.items()):
        trials.sort(key=lambda t: t["song"])
        for i, t in enumerate(trials):
            t["key"] = f"x{i:03d}"
        write(out / f"{sub}_silence.npz", trials)
        log.info("%s -> %d silence trials", sub, len(trials))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.prep.nmed")
    ap.add_argument("what", choices=["t", "h", "t-raw"])
    ap.add_argument("--versions", default=None, type=lambda s: s.split(","),
                    help="NMED-H stimulus versions to read (default: all four)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return {"t": cmd_t, "h": cmd_h, "t-raw": cmd_t_raw}[args.what](args)


if __name__ == "__main__":
    sys.exit(main())
