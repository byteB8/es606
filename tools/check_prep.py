"""Sanity-check preprocessed EEG and audio features: rates, lengths, alignment."""
import json
import numpy as np
from eg606.paths import derived_dir

root = derived_dir("musin_g")
f = dict(np.load(root / "audio_features.npz"))
print("audio fs:", f["fs"])
for i in (1, 6, 12):
    e, o = f[f"env{i:02d}"], f[f"onset{i:02d}"]
    print(f"  song {i:2d}: env n={len(e)} ({len(e)/64:.1f}s) mean={e.mean():.3f} std={e.std():.3f}"
          f" | onset std={o.std():.4f}")

subs = sorted(root.glob("sub-*.npz"))
print(f"\n{len(subs)} subjects cached")
d = np.load(subs[0], allow_pickle=True)
meta = json.loads(str(d["meta"]))
print("channels:", len(d["ch_names"]), "| first:", list(d["ch_names"][:5]))
print("block order:", [t["song"] for t in meta])
print("max duration-match error:", max(t["match_err"] for t in meta))
for t in meta[:4]:
    s = t["song"]
    eeg = d[f"song{s:02d}"]
    pre = d[f"pre{s:02d}"]
    env = f[f"env{s:02d}"]
    print(f"  song {s:2d}: eeg {eeg.shape} ({eeg.shape[1]/64:.1f}s) env {len(env)}"
          f" diff={eeg.shape[1]-len(env)} samples | silence {pre.shape}"
          f" | eeg uV std={eeg.std():.2e}")
