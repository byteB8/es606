"""NMED-T event markers and subject indexing; NMED-H repetition structure."""
import zipfile, io, tempfile
import h5py, numpy as np, scipy.io as sio
from eg606.paths import raw_dir

t, h = raw_dir("nmed_t"), raw_dir("nmed_h")

print("=== NMED-T song21_Imputed: who are the 20 slices?")
m = sio.loadmat(str(t / "song21_Imputed.mat"), squeeze_me=True)
subs = np.asarray(m["subs21"]).ravel()
print("  subs21:", [str(s) for s in subs])
print("  fs:", np.asarray(m["fs"]).ravel(), " data:", m["data21"].shape)

print("=== NMED-T 02_1_raw: DIN_1 event markers")
with h5py.File(t / "02_1_raw.mat", "r") as f:
    din = f["DIN_1"]
    print("  DIN_1 shape", din.shape)
    rows = []
    for c in range(din.shape[0]):
        row = []
        for r in range(din.shape[1]):
            o = f[din[c, r]]
            a = np.asarray(o)
            row.append("".join(chr(x) for x in a.ravel()) if a.dtype.kind in "iu" and a.size <= 8
                       and a.max() < 128 and a.min() > 31 else a.ravel()[:3])
        rows.append(row)
    for r in rows[:12]:
        print("   ", r)
    print("    ...")
    for r in rows[-4:]:
        print("   ", r)
    print("  X:", f["X"].shape, "fs:", np.asarray(f["fs"]).ravel())

print("=== NMED-H one clean file")
with zipfile.ZipFile(h / "CleanEEG_aggregatedByStimulus_orig.zip") as z:
    name = "CleanEEG_aggregatedByStimulus_orig/song21_a_Imputed.mat"
    with tempfile.NamedTemporaryFile(suffix=".mat") as tmp:
        with z.open(name) as src:
            tmp.write(src.read())
        tmp.flush()
        mm = sio.loadmat(tmp.name, squeeze_me=True)
        for k, v in mm.items():
            if k.startswith("__"):
                continue
            a = np.asarray(v)
            print(f"  {k}: {a.shape} {a.dtype}")
            if a.dtype == object or a.size < 60:
                print("     ", [str(x) for x in a.ravel()[:60]])
