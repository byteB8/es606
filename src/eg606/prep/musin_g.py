"""Preprocess MUSIN-G from sourcedata into per-subject epochs aligned to the audio.

Facts established by inspection (tools/explore_musing*.py):
  * sourcedata holds one continuous raw recording per subject (250 or 1000 Hz, 129 channels);
  * its events.tsv marks every trial with `stm+` (song onset) and `fxnd` (song end);
  * song identity is recovered from the trial duration, which matches one stimulus wav to <0.1 s;
  * per-session files under sub-XXX/ses-NN carry copied annotations and cannot be used for onsets.

    python -m eg606.prep.musin_g --subjects all
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

from eg606.audio.features import envelope, onset_envelope, read_wav
from eg606.paths import derived_dir, raw_dir

log = logging.getLogger(__name__)

FS_OUT = 64.0        # Hz, matches the SparrKULee derivative
BAND = (0.5, 32.0)   # Hz
PRE_SILENCE = 10.0   # s of silence before each song (control N1)
ICA_COMPONENTS = 40

# EGI outer ring: cheek/neck sensors dominated by EMG and eye movement
SUFFIX = {"biosemi64": "bs64"}

# Cleaning levels. The literature disagrees about how much cleaning helps decoding
# (Delorme 2023 "EEG is better left alone"; Del Pup et al. 2025), so the level is an
# experimental variable rather than a fixed choice:
#   none  - filter + average reference only
#   basic - + bad-channel detection (LOF) and spherical interpolation  [the steps both papers back]
#   ica   - + ICA with ICLabel rejection of eye/muscle/heart/line components
CLEAN_LEVELS = ("none", "basic", "ica")
ICLABEL_REJECT = {"eye blink", "muscle artifact", "heart beat", "line noise", "channel noise"}
ICLABEL_P = 0.8

RIM = ["E43", "E48", "E49", "E56", "E63", "E68", "E73", "E81", "E88", "E94",
       "E99", "E107", "E113", "E119", "E120", "E125", "E126", "E127", "E128"]


def song_durations(root: Path) -> dict[int, float]:
    out = {}
    for i in range(1, 13):
        x, sr = read_wav(root / "Code" / "ESongs" / f"{i}.esh.wav")
        out[i] = len(x) / sr
    return out


def trials(root: Path, sub: str, durations: dict[int, float]) -> list[dict]:
    """(song, onset, offset) per trial, song identified by matching trial length to a stimulus."""
    ev = glob.glob(str(root / "sourcedata" / sub / "eeg" / "*events.tsv"))[0]
    rows = [l.split("\t") for l in open(ev).read().splitlines()[1:]]
    onsets = [(float(r[0]), r[-1]) for r in rows]
    starts = [o for o, v in onsets if v == "stm+"]
    ends = [o for o, v in onsets if v == "fxnd"]
    out = []
    for pos, (st, en) in enumerate(zip(starts, ends), start=1):
        d = en - st
        song = min(durations, key=lambda k: abs(d - durations[k]))
        err = abs(d - durations[song])
        if err > 0.5:
            raise ValueError(f"{sub} trial {pos}: no stimulus matches duration {d:.2f}s")
        out.append({"song": song, "onset": st, "duration": durations[song],
                    "block_pos": pos, "match_err": err})
    songs = [t["song"] for t in out]
    if len(set(songs)) != 12:
        raise ValueError(f"{sub}: expected 12 distinct songs, got {sorted(songs)}")
    return out


def preprocess_subject(sub: str, root: Path, durations: dict[int, float],
                       to_montage: str | None = None, clean: str = "none") -> dict:
    setf = glob.glob(str(root / "sourcedata" / sub / "eeg" / "*_eeg.set"))[0]
    raw = mne.io.read_raw_eeglab(setf, preload=True, verbose="ERROR")

    # positions are needed for interpolation of bad channels and for montage mapping
    if "E129" in raw.ch_names:
        raw.rename_channels({"E129": "Cz"})
    raw.set_montage("GSN-HydroCel-129", on_missing="ignore", verbose="ERROR")
    ref = [c for c in raw.ch_names if c in ("E129", "Cz")]
    raw.drop_channels([c for c in RIM + ref if c in raw.ch_names])

    report = {"bads": [], "ica_excluded": 0}
    if clean in ("basic", "ica"):
        probe = raw.copy().filter(1.0, 45.0, fir_design="firwin", verbose="ERROR")
        raw.info["bads"] = mne.preprocessing.find_bad_channels_lof(probe, verbose="ERROR")
        report["bads"] = list(raw.info["bads"])
        if raw.info["bads"]:
            raw.interpolate_bads(reset_bads=True, verbose="ERROR")
        del probe

    if clean == "ica":
        # ICLabel expects broadband, average-referenced data and an extended-infomax fit
        ica_raw = (raw.copy().resample(250.0, npad="auto", verbose="ERROR")
                   .filter(1.0, 45.0, fir_design="firwin", verbose="ERROR")
                   .set_eeg_reference("average", verbose="ERROR"))
        ica = mne.preprocessing.ICA(n_components=ICA_COMPONENTS, method="infomax",
                                    fit_params=dict(extended=True), max_iter="auto",
                                    random_state=0, verbose="ERROR")
        ica.fit(ica_raw, verbose="ERROR")
        from mne_icalabel import label_components
        labels = label_components(ica_raw, ica, method="iclabel")
        ica.exclude = [i for i, (lab, prob) in enumerate(zip(labels["labels"], labels["y_pred_proba"]))
                       if lab in ICLABEL_REJECT and prob >= ICLABEL_P]
        report["ica_excluded"] = len(ica.exclude)
        report["ica_labels"] = [labels["labels"][i] for i in ica.exclude]
        ica.apply(raw, verbose="ERROR")
        del ica_raw

    raw.filter(*BAND, fir_design="firwin", verbose="ERROR")
    raw.set_eeg_reference("average", verbose="ERROR")
    raw.resample(FS_OUT, npad="auto", verbose="ERROR")
    if to_montage:
        # spherical-spline interpolation onto the speech cap, so one decoder fits both datasets
        raw = raw.interpolate_to(mne.channels.make_standard_montage(to_montage), method="MNE")

    data, meta = {}, []
    for t in trials(root, sub, durations):
        a = int(round(t["onset"] * FS_OUT))
        b = a + int(round(t["duration"] * FS_OUT))
        pre = int(round(PRE_SILENCE * FS_OUT))
        if b > raw.n_times or a - pre < 0:
            raise ValueError(f"{sub}: trial {t['block_pos']} outside the recording")
        data[f"song{t['song']:02d}"] = raw.get_data(start=a, stop=b).astype(np.float32)
        data[f"pre{t['song']:02d}"] = raw.get_data(start=a - pre, stop=a).astype(np.float32)
        meta.append(t)
    return {"data": data, "meta": meta, "ch_names": raw.ch_names, "report": report}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.prep.musin_g")
    ap.add_argument("--subjects", default="all", help="'all' or e.g. sub-001,sub-002")
    ap.add_argument("--force", action="store_true", help="redo subjects already cached")
    ap.add_argument("--clean", default="none", choices=CLEAN_LEVELS,
                    help="artifact handling level (an experimental variable, see CLEAN_LEVELS)")
    ap.add_argument("--to-montage", default=None,
                    help="interpolate onto a standard montage, e.g. biosemi64 (for transfer)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root, out = raw_dir("musin_g"), derived_dir("musin_g")
    out.mkdir(parents=True, exist_ok=True)
    durations = song_durations(root)

    # audio features, once for all subjects
    feat_path = out / "audio_features.npz"
    if args.force or not feat_path.exists():
        feats = {}
        for i in range(1, 13):
            x, sr = read_wav(root / "Code" / "ESongs" / f"{i}.esh.wav")
            feats[f"env{i:02d}"] = envelope(x, sr, FS_OUT).astype(np.float32)
            feats[f"onset{i:02d}"] = onset_envelope(x, sr, FS_OUT).astype(np.float32)
        np.savez_compressed(feat_path, fs=FS_OUT, **feats)
        log.info("audio features -> %s", feat_path)

    subs = (sorted(Path(p).name for p in glob.glob(str(root / "sourcedata" / "sub-*")))
            if args.subjects == "all" else args.subjects.split(","))
    for sub in subs:
        suffix = ("" if args.clean == "none" else f"_{args.clean}")
        suffix += f"_{SUFFIX[args.to_montage]}" if args.to_montage else ""
        dest = out / f"{sub}{suffix}.npz"
        if dest.exists() and not args.force:
            log.info("%s cached", sub); continue
        log.info("%s ...", sub)
        r = preprocess_subject(sub, root, durations, args.to_montage, args.clean)
        np.savez_compressed(dest, fs=FS_OUT, ch_names=np.array(r["ch_names"]),
                            meta=json.dumps(r["meta"]), report=json.dumps(r["report"]),
                            **r["data"])
        log.info("%s -> %s (%d channels, %d bad, %d ICs removed)", sub, dest.name,
                 len(r["ch_names"]), len(r["report"]["bads"]), r["report"]["ica_excluded"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
