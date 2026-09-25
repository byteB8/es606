"""Bach 'music of silence' (Marion, Di Liberto et al. 2021): EEG + our onset feature, aligned.

The release gives, per trial, 64 Hz EEG and the authors' note-onset signal already aligned to it.
To compare latencies across datasets the *same* acoustic feature must be used everywhere, so we
recover each trial's chorale and its time offset by cross-correlating the authors' onset train
against the spectral flux of each chorale's audio (right channel; the left carries the metronome),
then take the flux from that aligned position.

    python -m eg606.prep.bach
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

from eg606.audio.features import envelope, onset_envelope
from eg606.paths import derived_dir, raw_dir

log = logging.getLogger(__name__)
FS = 64
CHORALES = ["chor-019", "chor-038", "chor-096", "chor-101"]
LISTEN = 0          # condition index: 0 = listening, 1 = imagery


def read_right_channel(path: Path) -> tuple[np.ndarray, int]:
    import wave
    with wave.open(str(path)) as w:
        n, ch, width, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    dtype = {2: np.int16, 4: np.int32}[width]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float64).reshape(-1, ch)
    x = x[:, 1] if ch > 1 else x[:, 0]            # right channel: the music, not the metronome
    return x / (np.abs(x).max() or 1.0), sr


def align(onsets: np.ndarray, flux: dict) -> tuple[str, int, float]:
    """Best (chorale, offset, correlation) matching a trial's onset train to the audio flux."""
    o = (onsets != 0).astype(np.float64)
    o = (o - o.mean()) / (o.std() or 1.0)
    best = (None, 0, -np.inf)
    for name, f in flux.items():
        f = (f - f.mean()) / (f.std() or 1.0)
        for off in range(0, max(1, len(f) - len(o) + 1)):
            seg = f[off: off + len(o)]
            if len(seg) < len(o):
                break
            r = float(np.dot(o, seg) / len(o))
            if r > best[2]:
                best = (name, off, r)
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.prep.bach")
    ap.add_argument("--condition", type=int, default=LISTEN)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = raw_dir("bach_silence")
    audio_dir = root / "original_stim" / "original_stim" / "audio"
    flux, env = {}, {}
    for c in CHORALES:
        x, sr = read_right_channel(audio_dir / f"{c}.wav")
        flux[c] = onset_envelope(x, sr, FS)
        env[c] = envelope(x, sr, FS)
        log.info("%s: %.1f s of audio", c, len(x) / sr)

    m = sio.loadmat(str(root / "ImageryData.mat"), squeeze_me=False)
    fs = int(np.asarray(m["downFs"]).ravel()[0])
    if fs != FS:
        raise ValueError(f"expected {FS} Hz, file says {fs}")
    eeg_all, stim_all = m["eeg"], m["stim"]
    out = derived_dir("bach_silence")
    out.mkdir(parents=True, exist_ok=True)

    report = []
    # nesting (verified with tools/probe_bach_cells.py):
    #   eeg[0, participant][0, condition][0, trial] -> (1803 time, 64 channels)
    for p in range(eeg_all.shape[1]):
        eeg_c = eeg_all[0, p][0, args.condition]
        stim_c = stim_all[0, p][0, args.condition]
        data, meta = {}, []
        for tr in range(eeg_c.shape[1]):
            e = np.asarray(eeg_c[0, tr], dtype=np.float32)                     # (time, channels)
            s = np.asarray(stim_c[0, tr], dtype=np.float64).ravel()
            if e.size == 0 or s.size == 0:
                continue
            name, off, r = align(s, flux)
            n = min(e.shape[0], len(s), len(flux[name]) - off)
            data[f"eeg{tr:02d}"] = e[:n].T                                     # (channels, time)
            data[f"onset{tr:02d}"] = flux[name][off: off + n].astype(np.float32)
            data[f"env{tr:02d}"] = env[name][off: off + n].astype(np.float32)
            meta.append({"trial": tr, "chorale": name, "offset": int(off), "align_r": r, "n": int(n)})
        np.savez_compressed(out / f"sub-{p + 1:02d}.npz", fs=FS, meta=json.dumps(meta), **data)
        rs = [x["align_r"] for x in meta]
        report.append((p + 1, len(meta), float(np.median(rs)), min(rs)))
        log.info("sub-%02d: %d trials, alignment r median %.2f min %.2f, chorales %s", p + 1, len(meta),
                 np.median(rs), min(rs), sorted({x['chorale'] for x in meta}))
    print("\nalignment quality (correlation of the authors' onsets with our audio flux):")
    print(f"  median of per-listener medians {np.median([r[2] for r in report]):.2f}; "
          f"worst trial {min(r[3] for r in report):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
