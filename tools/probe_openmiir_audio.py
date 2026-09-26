"""OpenMIIR stimuli: durations, cue length, and whether our feature code handles them."""
from pathlib import Path

import numpy as np

from eg606.audio.features import envelope, onset_envelope, read_wav
from eg606.paths import raw_dir

root = raw_dir("openmiir")
for version in ("v1", "v2"):
    full = sorted((root / "audio" / f"full.{version}").glob("*.wav"))
    if not full:
        continue
    print(f"=== full.{version}: {len(full)} stimuli")
    for p in full:
        x, sr = read_wav(p)
        stim_id = p.name.split("_")[0].lstrip("S")
        cue = root / "audio" / f"cues.{version}" / p.name.replace(".wav", "_cue.wav")
        cue_s = None
        if cue.exists():
            c, csr = read_wav(cue)
            cue_s = len(c) / csr
        beats = root / "meta" / f"beats.{version}" / f"{int(stim_id)}_beats.txt"
        first_beat = None
        if beats.exists():
            vals = [float(v) for v in beats.read_text().split()]
            first_beat = vals[0] if vals else None
        env = envelope(x, sr, 64.0)
        ons = onset_envelope(x, sr, 64.0)
        print(f"  {p.name[:44]:<44} {len(x) / sr:6.2f} s  sr={sr}  "
              f"cue={cue_s if cue_s is None else round(cue_s, 2)}  "
              f"first_beat={first_beat}  env={len(env)} onset={len(ons)}")
