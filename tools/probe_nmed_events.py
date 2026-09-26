"""Which NMED-T trigger code marks a song onset? Codes that occur ~5x per session are candidates."""
from collections import Counter
import h5py, numpy as np
from eg606.paths import raw_dir

t = raw_dir("nmed_t")


def events(path):
    with h5py.File(path, "r") as f:
        din = f["DIN_1"]
        out = []
        for c in range(din.shape[0]):
            code = "".join(chr(x) for x in np.asarray(f[din[c, 0]]).ravel())
            onset = float(np.asarray(f[din[c, 1]]).ravel()[0])
            out.append((code, onset))
        return out, f["X"].shape, float(np.asarray(f["fs"]).ravel()[0])


for name in ("02_1_raw.mat", "02_2_raw.mat", "03_1_raw.mat"):
    ev, shape, fs = events(t / name)
    c = Counter(code for code, _ in ev)
    print(f"=== {name}  X={shape} fs={fs:.0f}  {len(ev)} events")
    print("   ", dict(sorted(c.items(), key=lambda kv: -kv[1])))
    for code, n in sorted(c.items()):
        if 3 <= n <= 6:
            gaps = np.diff([o for cc, o in ev if cc == code]) / fs
            print(f"    {code} x{n}: onsets {[round(o / fs, 1) for cc, o in ev if cc == code]}"
                  f"  gaps {[round(g, 1) for g in gaps]}")

print("\n=== time-ordered event sequence, 02_1 and 02_2")
for name in ("02_1_raw.mat", "02_2_raw.mat"):
    ev, shape, fs = events(t / name)
    print(f"--- {name} (recording {shape[0] / fs:.0f} s)")
    for code, o in sorted(ev, key=lambda x: x[1]):
        if code != "D128":
            print(f"    {o / fs:8.1f}s  {code}")
