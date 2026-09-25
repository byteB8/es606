"""One-off: what does MUSIN-G actually look like on disk?"""
import glob, os, sys
import mne
from eg606.paths import raw_dir

root = raw_dir("musin_g")
print("root:", root)
print("\n--- top level"); print(sorted(os.listdir(root))[:12])
print("\n--- sub-001 tree")
for p in sorted(glob.glob(str(root / "sub-001" / "*" / "*" / "*")))[:8]:
    print("   ", os.path.relpath(p, root), os.path.getsize(p) // 1024, "KB")

ev = sorted(glob.glob(str(root / "sub-001" / "*" / "eeg" / "*events.tsv")))
print(f"\n--- events files for sub-001: {len(ev)}")
for f in ev[:3]:
    print("\n", os.path.basename(f))
    print(open(f).read()[:600])

chan = sorted(glob.glob(str(root / "sub-001" / "*" / "eeg" / "*channels.tsv")))
if chan:
    lines = open(chan[0]).read().splitlines()
    print(f"\n--- channels.tsv: {len(lines)-1} channels")
    print("\n".join(lines[:4]))

setf = sorted(glob.glob(str(root / "sub-001" / "*" / "eeg" / "*_eeg.set")))
print(f"\n--- .set files for sub-001: {len(setf)}")
if setf:
    raw = mne.io.read_raw_eeglab(setf[0], preload=False, verbose="ERROR")
    print("first:", os.path.basename(setf[0]))
    print("   sfreq", raw.info["sfreq"], "nchan", raw.info["nchan"],
          "duration_s", round(raw.n_times / raw.info["sfreq"], 1))
    print("   ch names[:8]", raw.ch_names[:8])
    try:
        ann = raw.annotations
        print("   annotations:", len(ann), set(ann.description) if len(ann) else "none")
    except Exception as e:
        print("   annotations error", e)

src = sorted(glob.glob(str(root / "sourcedata" / "sub-001" / "eeg" / "*")))
print("\n--- sourcedata sub-001:", [os.path.basename(p) for p in src])
for f in src:
    if f.endswith("events.tsv"):
        txt = open(f).read().splitlines()
        print(f"   {os.path.basename(f)}: {len(txt)-1} rows")
        print("   " + "\n   ".join(txt[:8]))

print("\n--- stimuli / Code")
for p in sorted(glob.glob(str(root / "Code" / "*")))[:5]:
    print("   ", os.path.relpath(p, root))
for p in sorted(glob.glob(str(root / "stimuli" / "*"))):
    print("   ", os.path.relpath(p, root))
