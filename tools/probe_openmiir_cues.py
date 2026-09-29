"""Cross-check the authors' CUE_LENGTH table against the cue wavs we downloaded.

The full stimulus wav is cue clicks followed by music, and the EEG marks the start of the *wav*,
so the music begins one cue-length later. That offset has to be right or every OpenMIIR alignment
is wrong, which is exactly the class of error this project is about.
"""
from eg606.audio.features import read_wav
from eg606.paths import raw_dir

# from deepthought/datasets/openmiir/constants.py (commented out there, v1 values)
CUE_LENGTH_V1 = {1: 1.649, 2: 1.865, 3: 2.315, 4: 2.986, 11: 1.668, 12: 1.897,
                 13: 2.362, 14: 2.973, 21: 1.995, 22: 2.046, 23: 2.306, 24: 3.372}

root = raw_dir("openmiir") / "audio"
for version in ("v1", "v2"):
    cues = sorted((root / f"cues.{version}").glob("*.wav"))
    full = {p.name.split("_")[0]: p for p in (root / f"full.{version}").glob("*.wav")}
    if not cues:
        continue
    print(f"=== cues.{version}: {len(cues)} files")
    for p in cues:
        sid = int(p.name.split("_")[0].lstrip("S"))
        c, sr = read_wav(p)
        cue_s = len(c) / sr
        f = full.get(p.name.split("_")[0])
        full_s = None
        if f:
            x, fsr = read_wav(f)
            full_s = len(x) / fsr
        ref = CUE_LENGTH_V1.get(sid)
        delta = "" if ref is None else f"  const={ref:.3f}  diff={cue_s - ref:+.3f}"
        music = "" if full_s is None else f"  full={full_s:6.2f}s  music={full_s - cue_s:6.2f}s"
        print(f"  stim {sid:>2}  cue={cue_s:6.3f}s{delta}{music}")
