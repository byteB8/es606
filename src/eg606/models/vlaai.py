"""Zero-shot probe: run the published VLAAI speech decoder on music, untouched.

VLAAI (Accou et al., Sci Rep 2023) is a subject-independent speech-envelope decoder trained on
144 h of speech-listening EEG, released as ONNX weights for 64-channel EEG at 64 Hz. Our music is
interpolated onto the same BioSemi-64 montage (verified identical channel order), so the model can
be applied to music with no retraining at all.

    python -m eg606.models.vlaai --dataset music --windows 5,10,30,60
"""
from __future__ import annotations

import argparse
import json
import re
import logging
import sys
from pathlib import Path

import numpy as np

from eg606.eval.linear import FS, match_mismatch, zscore
from eg606.paths import DATA_ROOT, derived_dir

log = logging.getLogger(__name__)

MODEL = DATA_ROOT / "models" / "vlaai.onnx"
CHUNK = 64 * 50          # 50 s, matching the released evaluation examples
TARGET_STD = 5.0         # their example EEG sits near this scale


def session(path: Path = MODEL):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def decode(sess, eeg: np.ndarray) -> np.ndarray:
    """EEG (channels, time) -> predicted envelope (time,), run in chunks."""
    x = zscore(eeg.astype(np.float32)) * TARGET_STD           # match the training scale
    name = sess.get_inputs()[0].name
    out = []
    for start in range(0, x.shape[1], CHUNK):
        block = x[:, start:start + CHUNK]
        if block.shape[1] < 64:                                # too short to be meaningful
            break
        y = sess.run(None, {name: block.T[None].astype(np.float32)})[0]
        out.append(np.asarray(y).reshape(-1))
    return np.concatenate(out) if out else np.zeros(0)


def music_trials(root: Path, feats: dict, kind: str):
    for p in sorted(x for x in root.glob("sub-*_bs64.npz") if re.match(r"^sub-\d+_bs64$", x.stem)):
        d = np.load(p, allow_pickle=True)
        trials = []
        for m in json.loads(str(d["meta"])):
            song = m["song"]
            eeg = d[f"song{song:02d}"].astype(np.float64)
            y = feats[f"{kind}{song:02d}"].astype(np.float64)
            n = min(eeg.shape[1], len(y))
            trials.append((eeg[:, :n], zscore(y[:n])))
        yield p.stem.split("_")[0], trials


def speech_trials(root: Path, feats: dict, kind: str, limit: int):
    for p in sorted(root.glob("sub-*.npz"))[:limit]:
        d = np.load(p, allow_pickle=True)
        trials = []
        for m in json.loads(str(d["meta"])):
            y = feats.get(f"{kind}::{m['stimulus']}")
            if y is None:
                continue
            eeg = d[m["key"]].astype(np.float64)
            n = min(eeg.shape[1], len(y))
            trials.append((eeg[:, :n], zscore(y[:n].astype(np.float64))))
        yield p.stem, trials


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.models.vlaai")
    ap.add_argument("--dataset", default="music", choices=["music", "speech"])
    ap.add_argument("--feature", default="env", choices=["env", "onset"])
    ap.add_argument("--windows", default="5,10,30,60")
    ap.add_argument("--limit", type=int, default=10, help="speech listeners to probe")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    windows = [float(w) for w in args.windows.split(",")]

    sess = session()
    log.info("loaded %s", MODEL)

    if args.dataset == "music":
        root = derived_dir("musin_g")
        feats = dict(np.load(root / "audio_features.npz"))
        source = music_trials(root, feats, args.feature)
    else:
        root = derived_dir("sparrkulee")
        feats = dict(np.load(root / "audio_features.npz"))
        source = speech_trials(root, feats, args.feature, args.limit)

    results = {}
    for sub, trials in source:
        rs, mm = [], {f"{x:g}s": [0, 0] for x in windows}
        for eeg, y in trials:
            pred = decode(sess, eeg)
            n = min(len(pred), len(y))
            if n < 64 or pred[:n].std() == 0:
                continue
            rs.append(np.corrcoef(pred[:n], y[:n])[0, 1])
            for x in windows:
                h, t = match_mismatch(pred[:n], y[:n], int(x * FS))
                mm[f"{x:g}s"][0] += h
                mm[f"{x:g}s"][1] += t
        results[sub] = {"r": float(np.mean(rs)) if rs else float("nan"),
                        "mm": {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in mm.items()}}
        log.info("%s  r=%+.4f  %s", sub, results[sub]["r"],
                 {k: round(v, 3) for k, v in results[sub]["mm"].items()})

    r = np.array([v["r"] for v in results.values()])
    print(f"\n=== VLAAI zero-shot on {args.dataset} ({len(r)} listeners, chance = 0.5)")
    print(f"envelope r: {np.nanmean(r):+.4f} +/- {np.nanstd(r):.4f}")
    for x in windows:
        k = f"{x:g}s"
        a = np.array([v["mm"][k] for v in results.values()])
        print(f"  {k:>4}: {np.nanmean(a):.3f} +/- {np.nanstd(a):.3f}  "
              f"({int((a > 0.5).sum())}/{len(a)} above chance)")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
