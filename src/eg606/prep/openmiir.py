"""Preprocess OpenMIIR perception trials into the common 64 Hz format.

OpenMIIR (Stober et al., ISMIR 2015) is the third music dataset here and the only one recorded on
the same cap as the speech data, so a speech-trained decoder can be applied to it **without any
montage interpolation**. That removes the one step in the speech->music comparison that has so far
gone unexamined.

Facts established by inspection (tools/probe_openmiir*.py), all from the authors' own code and
metadata rather than assumed:
  * trial events are `stimulus_id * 10 + condition`, with condition 1 = perception and 2-4
    imagination (deepthought/datasets/openmiir/events.py). Only perception is used here.
  * event code 1000 is a dedicated **audio onset** marker, which the authors describe as "the more
    precise audio onset marker" and use to replace the trial trigger. This project exists partly
    because MUSIN-G has no such marker, so where OpenMIIR provides one it is used, and the
    trigger-to-audio latency it reveals is reported.
  * each stimulus wav is cue clicks *overlapping* the start of the music, not concatenated with it,
    so the music does not begin at the end of the cue wav. `Stimuli_Meta.v{1,2}.xlsx` gives the
    authoritative offset in its "length of cue (sec)" column, measured from the first click to the
    last; its "length of cue only" column matches the standalone cue wav to the millisecond, which
    is what ties the table to these files.
  * stimulus version v1 was presented to P01-P08 and v2 to P09-P14. Only 5 of the 12 v1 stimuli
    were released as audio, so v1 listeners contribute those 5 and v2 listeners all 12. The two
    versions differ slightly in length, so a missing v1 file is skipped rather than substituted
    with its v2 counterpart, which would reintroduce exactly the alignment error this project is
    about.

    python -m eg606.prep.openmiir --subjects all
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

from eg606.audio.features import envelope, onset_envelope, read_wav
from eg606.paths import derived_dir, raw_dir

log = logging.getLogger(__name__)

FS_OUT = 64.0
BAND = (0.5, 32.0)
PERCEPTION = 1            # the listening condition; 2-4 are imagination
AUDIO_ONSET = 1000        # the authors' precise onset marker
NOISE = 1111
MONTAGE = "biosemi64"
# v1 was presented to P01-P08, v2 to P09-P14
V2_FROM = 9


def stimulus_meta(root: Path, version: str) -> dict[int, dict]:
    """{stimulus_id: {cue, song, file}} in seconds, from the authors' workbook."""
    import openpyxl

    wb = openpyxl.load_workbook(root / "meta" / f"Stimuli_Meta.{version}.xlsx", data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(c).strip() if c is not None else "" for c in rows[0]]
    col = {name: header.index(name) for name in
           ("id", "audio file", "length of cue (sec)", "length of song (sec)")}
    out = {}
    for row in rows[1:]:
        if not isinstance(row[col["id"]], (int, float)):
            continue
        out[int(row[col["id"]])] = {
            "file": str(row[col["audio file"]]),
            "cue": float(row[col["length of cue (sec)"]]),
            "song": float(row[col["length of song (sec)"]]),
        }
    return out


def audio_features(root: Path, version: str, meta: dict[int, dict]) -> dict[str, np.ndarray]:
    """Envelope and onset flux of the *music* portion of each released stimulus, at FS_OUT."""
    feats, missing = {}, []
    for sid, m in meta.items():
        path = root / "audio" / f"full.{version}" / m["file"]
        if not path.exists():
            missing.append(sid)
            continue
        x, sr = read_wav(path)
        music = x[int(round(m["cue"] * sr)):]
        feats[f"env{sid:02d}"] = envelope(music, sr, FS_OUT).astype(np.float32)
        feats[f"onset{sid:02d}"] = onset_envelope(music, sr, FS_OUT).astype(np.float32)
    if missing:
        log.warning("%s: no audio released for stimuli %s; those trials are dropped",
                    version, missing)
    return feats


def find_stim_channel(raw: mne.io.BaseRaw) -> str:
    """Whatever the trigger channel is called here: MNE's 'STI 014' or BioSemi's 'Status'."""
    stim = [raw.ch_names[i] for i in mne.pick_types(raw.info, stim=True, meg=False, eeg=False)]
    for name in ("STI 014", "Status", "STI101"):
        if name in raw.ch_names:
            return name
    if not stim:
        raise ValueError(f"no stim channel among {raw.ch_names[:8]}...")
    return stim[0]


def trials(raw: mne.io.BaseRaw, meta: dict[int, dict]):
    """Perception trials, timed from the audio-onset marker where the recording provides one."""
    events = mne.find_events(raw, stim_channel=find_stim_channel(raw),
                             shortest_event=1, verbose="ERROR")
    fs = raw.info["sfreq"]
    out, latencies, missing = [], [], 0
    for i, (sample, _, code) in enumerate(events):
        if code >= AUDIO_ONSET or code % 10 != PERCEPTION:
            continue
        sid = int(code) // 10
        if sid not in meta:
            continue
        onset = sample
        if i + 1 < len(events) and events[i + 1][2] == AUDIO_ONSET:
            onset = events[i + 1][0]
            latencies.append(float((onset - sample) / fs * 1000))
        else:
            missing += 1
        out.append({"stimulus": int(sid), "trigger": int(sample), "audio_onset": int(onset),
                    "trigger_to_audio_ms": float((onset - sample) / fs * 1000),
                    "music_onset": onset + meta[sid]["cue"] * fs,
                    "duration": meta[sid]["song"]})
    return out, latencies, missing


def preprocess(path: Path, meta: dict[int, dict], available: set[int]) -> dict:
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    tr, latencies, missing = trials(raw, meta)
    tr = [t for t in tr if t["stimulus"] in available]

    raw.pick("eeg")
    montage = mne.channels.make_standard_montage(MONTAGE)
    keep = [c for c in raw.ch_names if c in montage.ch_names]
    if len(keep) < 64:
        # BioSemi files often keep the amplifier's own labels (A1-A32, B1-B32). Those are the
        # same 64 positions in the same order as the standard cap, so rename by position rather
        # than dropping every channel. Anything beyond the first 64 is EXG/auxiliary.
        amp = [c for c in raw.ch_names if re.fullmatch(r"[AB]\d{1,2}", c)]
        if len(amp) >= 64:
            raw.pick(amp[:64])
            raw.rename_channels(dict(zip(amp[:64], montage.ch_names)))
            keep = list(montage.ch_names)
            log.info("renamed BioSemi amplifier labels (%s...) to the standard cap", amp[:3])
        else:
            raise ValueError(f"cannot map {len(keep)} channels onto {MONTAGE}: {raw.ch_names[:8]}")
    raw.pick(keep).set_montage(montage, verbose="ERROR")
    raw.filter(*BAND, fir_design="firwin", verbose="ERROR")
    raw.set_eeg_reference("average", verbose="ERROR")
    ratio = FS_OUT / raw.info["sfreq"]
    raw.resample(FS_OUT, npad="auto", verbose="ERROR")

    data, kept = {}, []
    for t in tr:
        a = int(round(t["music_onset"] * ratio))
        b = a + int(round(t["duration"] * FS_OUT))
        if a < 0 or b > raw.n_times:
            continue
        key = f"x{len(kept):03d}"
        data[key] = raw.get_data(start=a, stop=b).astype(np.float32)
        kept.append({"key": key, "stimulus": int(t["stimulus"]), "song": int(t["stimulus"]),
                     "trigger_to_audio_ms": float(t["trigger_to_audio_ms"])})
    return {"data": data, "meta": kept, "ch_names": raw.ch_names,
            "latencies_ms": latencies, "missing_audio_onset": missing}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.prep.openmiir")
    ap.add_argument("--subjects", default="all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root, out = raw_dir("openmiir"), derived_dir("openmiir")
    out.mkdir(parents=True, exist_ok=True)
    fifs = sorted((root / "eeg").glob("P*-raw.fif"))
    if args.subjects != "all":
        want = set(args.subjects.split(","))
        fifs = [p for p in fifs if p.stem.split("-")[0] in want]
    if not fifs:
        log.error("no P*-raw.fif under %s — the raw EEG is only distributed by BitTorrent, "
                  "see eg606.data.download.openmiir", root / "eeg")
        return 1

    metas = {v: stimulus_meta(root, v) for v in ("v1", "v2")}
    available = {}
    for v, m in metas.items():
        feat_path = out / f"audio_features_{v}.npz"
        if args.force or not feat_path.exists():
            feats = audio_features(root, v, m)
            np.savez_compressed(feat_path, fs=FS_OUT, **feats)
            log.info("audio features %s -> %s (%d stimuli)", v, feat_path.name, len(feats) // 2)
        keys = set(np.load(feat_path).files)
        available[v] = {sid for sid in m if f"onset{sid:02d}" in keys}
        log.info("%s: %d stimuli with audio %s", v, len(available[v]), sorted(available[v]))

    all_lat = []
    for p in fifs:
        sub = p.stem.split("-")[0]
        version = "v2" if int(sub[1:]) >= V2_FROM else "v1"
        dest = out / f"sub-{sub}.npz"
        if dest.exists() and not args.force:
            log.info("%s cached", sub)
            continue
        r = preprocess(p, metas[version], available[version])
        np.savez_compressed(dest, fs=FS_OUT, version=version,
                            ch_names=np.array(r["ch_names"]),
                            meta=json.dumps(r["meta"]), **r["data"])
        all_lat += r["latencies_ms"]
        log.info("%s (%s): %d perception trials, %d channels, %d without an audio-onset marker",
                 sub, version, len(r["meta"]), len(r["ch_names"]), r["missing_audio_onset"])

    if all_lat:
        a = np.asarray(all_lat)
        print(f"\ntrigger -> audio-onset latency across {len(a)} perception trials: "
              f"median {np.median(a):.1f} ms, IQR {np.percentile(a, 25):.1f}-"
              f"{np.percentile(a, 75):.1f}, range {a.min():.1f}-{a.max():.1f}")
        print("This is the correction MUSIN-G does not provide; see the lab notebook.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
