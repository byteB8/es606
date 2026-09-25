"""Find the session -> song mapping and the song onset marker."""
import glob, os, collections
import mne, numpy as np
from eg606.paths import raw_dir

root = raw_dir("musin_g")
print("=== README"); print(open(root / "README").read()[:1500])

f = sorted(glob.glob(str(root / "sub-001" / "ses-01" / "eeg" / "*events.tsv")))[0]
rows = [l.split("\t") for l in open(f).read().splitlines()]
hdr = rows[0]
print("\n=== events.tsv value counts (sub-001 ses-01):")
print(collections.Counter(r[-1] for r in rows[1:]))
print("non-CELL rows:")
for r in rows[1:]:
    if r[-1] not in ("CELL",):
        print("   onset", round(float(r[0]), 2), "value", r[-1])

print("\n=== annotations per session, sub-001")
for s in range(1, 13):
    p = glob.glob(str(root / "sub-001" / f"ses-{s:02d}" / "eeg" / "*_eeg.set"))
    if not p: continue
    raw = mne.io.read_raw_eeglab(p[0], preload=False, verbose="ERROR")
    dur = raw.n_times / raw.info["sfreq"]
    ann = raw.annotations
    marks = [(d, round(o, 2)) for d, o in zip(ann.description, ann.onset) if d not in ("CELL",)]
    print(f" ses-{s:02d} dur={dur:7.1f}s  n_ann={len(ann):3d}  {marks[:6]}")

print("\n=== Song_Description")
print(open(root / "stimuli" / "Song_Description").read()[:1200])
print("\n=== Behavioural_data head")
print(open(root / "stimuli" / "Behavioural_data").read()[:800])
