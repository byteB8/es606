"""What is inside the Bach 'music of silence' dataset?"""
import os, zipfile
import numpy as np
from eg606.paths import raw_dir

root = raw_dir("bach_silence")
print("files:", sorted(os.listdir(root)))
print("\n=== README.txt")
print(open(root / "README.txt").read())

stim_dir = root / "original_stim"
if stim_dir.exists():
    for dp, dn, fn in os.walk(stim_dir):
        for f in sorted(fn)[:20]:
            print("  stim:", os.path.relpath(os.path.join(dp, f), root), os.path.getsize(os.path.join(dp, f)) // 1024, "KB")

mat = root / "ImageryData.mat"
print("\n=== ImageryData.mat:", os.path.getsize(mat) // 1_000_000, "MB")
try:
    import scipy.io as sio
    m = sio.loadmat(str(mat), squeeze_me=False, struct_as_record=False)
    for k, v in m.items():
        if k.startswith("__"):
            continue
        print(f"  {k}: type={type(v).__name__} shape={getattr(v, 'shape', None)} dtype={getattr(v, 'dtype', None)}")
except NotImplementedError:
    import h5py
    with h5py.File(mat, "r") as f:
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape} dtype={obj.dtype}")
        f.visititems(show)
