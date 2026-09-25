"""Assemble SparrKULee speech trials: published 64 Hz EEG + our own audio features.

The released derivative is already 64 Hz, high-passed at 0.5 Hz, blink-cleaned and average
referenced, so no EEG filtering is repeated here. The audio features are recomputed with the same
code used for music, which is what makes the speech->music comparison meaningful.

    python -m eg606.prep.sparrkulee --subjects all
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

from eg606.audio.features import envelope, onset_envelope
from eg606.paths import derived_dir, raw_dir

log = logging.getLogger(__name__)

FS_OUT = 64.0
STIM_RE = re.compile(r"desc-preproc-audio-(.+)_eeg\.npy$")


def stimulus_of(path: str) -> str:
    m = STIM_RE.search(Path(path).name)
    if not m:
        raise ValueError(f"cannot read the stimulus name from {path}")
    return m.group(1)


def audio_features(root: Path, stim: str) -> dict[str, np.ndarray]:
    with gzip.open(root / "stimuli" / "eeg" / f"{stim}.npz.gz") as fh:
        z = np.load(fh, allow_pickle=True)
        x, sr = z["audio"].astype(np.float64), int(z["fs"])
    x = x / (np.abs(x).max() or 1.0)
    return {"env": envelope(x, sr, FS_OUT).astype(np.float32),
            "onset": onset_envelope(x, sr, FS_OUT).astype(np.float32)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.prep.sparrkulee")
    ap.add_argument("--subjects", default="all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--features-only", action="store_true", help="build the audio cache and stop")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root, out = raw_dir("sparrkulee"), derived_dir("sparrkulee")
    out.mkdir(parents=True, exist_ok=True)
    eeg_root = root / "derivatives" / "preprocessed_eeg"

    runs = sorted(glob.glob(str(eeg_root / "sub-*" / "*" / "*.npy")))
    stimuli = sorted({stimulus_of(r) for r in runs})
    log.info("%d runs, %d distinct stimuli", len(runs), len(stimuli))

    # audio features: once per stimulus, shared by every listener who heard it
    feat_path = out / "audio_features.npz"
    feats = dict(np.load(feat_path)) if feat_path.exists() and not args.force else {}
    missing = [s for s in stimuli if f"env::{s}" not in feats]
    if missing:
        for i, s in enumerate(missing, 1):
            f = audio_features(root, s)
            feats[f"env::{s}"] = f["env"]
            feats[f"onset::{s}"] = f["onset"]
            if i % 20 == 0 or i == len(missing):
                log.info("audio features %d/%d", i, len(missing))
        np.savez_compressed(feat_path, fs=FS_OUT, **feats)
        log.info("audio features -> %s", feat_path)
    if args.features_only:
        return 0

    subs = sorted({Path(r).parts[-3] for r in runs})
    if args.subjects != "all":
        keep = set(args.subjects.split(","))
        subs = [s for s in subs if s in keep]

    for sub in subs:
        dest = out / f"{sub}.npz"
        if dest.exists() and not args.force:
            log.info("%s cached", sub); continue
        data, meta = {}, []
        for r in sorted(glob.glob(str(eeg_root / sub / "*" / "*.npy"))):
            stim = stimulus_of(r)
            eeg = np.load(r)
            if eeg.ndim != 2 or eeg.shape[0] != 64:
                log.warning("%s: unexpected shape %s, skipped", Path(r).name, eeg.shape)
                continue
            key = f"run{len(meta):02d}"
            data[key] = np.nan_to_num(eeg).astype(np.float32)
            meta.append({"key": key, "stimulus": stim, "n": int(eeg.shape[1])})
        if not meta:
            log.warning("%s: no usable runs", sub); continue
        np.savez_compressed(dest, fs=FS_OUT, meta=json.dumps(meta), **data)
        log.info("%s -> %d runs", sub, len(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
