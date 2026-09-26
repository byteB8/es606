"""What is inside the NMED-T/-H releases? Establishes the layout prep/nmed.py relies on.

The release mixes MATLAB v5 (scipy.io) and v7.3/HDF5 (h5py) files, so both readers are needed.
"""
import zipfile

import h5py
import numpy as np
import scipy.io as sio

from eg606.paths import raw_dir


def describe(path, limit=12):
    try:
        m = sio.loadmat(str(path), squeeze_me=False)
        for k, v in list(m.items())[:limit]:
            if k.startswith("__"):
                continue
            a = np.asarray(v)
            print(f"  v5 {k}: {a.shape} {a.dtype}")
            if a.size < 40 and a.dtype.kind in "fiu":
                print("       ", a.ravel()[:40])
        return
    except (ValueError, NotImplementedError):
        pass
    with h5py.File(path, "r") as f:
        def walk(g, prefix="", depth=0):
            for k, v in list(g.items())[:limit]:
                if isinstance(v, h5py.Dataset):
                    print(f"  h5 {prefix}{k}: {v.shape} {v.dtype}")
                    if v.size < 40 and v.dtype.kind in "fiu":
                        print("       ", np.asarray(v).ravel()[:40])
                elif depth < 2:
                    print(f"  h5 {prefix}{k}/")
                    walk(v, prefix + k + "/", depth + 1)
        walk(f)


t, h = raw_dir("nmed_t"), raw_dir("nmed_h")
for p in (t / "song21_Imputed.mat", t / "02_1_raw.mat", t / "participantInfo.mat",
          h / "stimAssignment.mat", h / "behaveStruct_a.mat"):
    print(f"=== {p.parent.name}/{p.name}")
    describe(p)
print("=== NMED-H CleanEEG_aggregatedByStimulus_orig.zip")
with zipfile.ZipFile(h / "CleanEEG_aggregatedByStimulus_orig.zip") as z:
    for i in z.infolist()[:16]:
        print(f"  {i.filename}  {i.file_size / 1e6:.0f} MB")
