"""Verify: does session number == song number, and where does the song start in each file?"""
import glob, wave, contextlib
import mne
from eg606.paths import raw_dir

root = raw_dir("musin_g")

wav_dur = {}
for i in range(1, 13):
    p = root / "Code" / "ESongs" / f"{i}.esh.wav"
    with contextlib.closing(wave.open(str(p))) as w:
        wav_dur[i] = w.getnframes() / w.getframerate()

print("song  wav_dur")
for i, d in wav_dur.items():
    print(f"  {i:2d}  {d:8.2f}")

for sub in ["sub-001", "sub-004", "sub-010"]:
    print(f"\n=== {sub}")
    print(" ses   file_dur  dur-10   wav(ses)   match?   annotations(onset)")
    for s in range(1, 13):
        p = glob.glob(str(root / sub / f"ses-{s:02d}" / "eeg" / "*_eeg.set"))
        if not p:
            print(f" {s:3d}   MISSING"); continue
        raw = mne.io.read_raw_eeglab(p[0], preload=False, verbose="ERROR")
        dur = raw.n_times / raw.info["sfreq"]
        ann = [(d, round(o, 2)) for d, o in zip(raw.annotations.description, raw.annotations.onset)
               if d in ("stm+", "fxcl", "fxnd", "bgin", "boundary", "stim")]
        w = wav_dur[s]
        ok = "YES" if abs((dur - 10) - w) < 1.5 else "no"
        # which song does the duration actually match best?
        best = min(wav_dur, key=lambda k: abs((dur - 10) - wav_dur[k]))
        print(f" {s:3d}  {dur:8.2f} {dur-10:8.2f}  {w:8.2f}   {ok} (best={best})  {ann[:4]}")
