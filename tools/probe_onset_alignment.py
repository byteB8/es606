"""Do SparrKULee runs start at stimulus onset? (needed to use them as an ERP latency reference)"""
import json
import numpy as np
from eg606.paths import derived_dir

root = derived_dir("sparrkulee")
feats = dict(np.load(root / "audio_features.npz"))
p = sorted(root.glob("sub-*.npz"))[0]
d = np.load(p, allow_pickle=True)
print(p.name)
for m in json.loads(str(d["meta"]))[:6]:
    env = feats.get(f"env::{m['stimulus']}")
    print(f"  {m['key']} {m['stimulus'][:34]:<34} eeg n={m['n']:6d}  audio n={len(env):6d}  diff={m['n']-len(env):+d}")

b = derived_dir("bach_silence")
q = sorted(b.glob("sub-*.npz"))[0]
db = np.load(q, allow_pickle=True)
mb = json.loads(str(db["meta"]))
print(q.name, "offsets:", sorted({x["offset"] for x in mb})[:12], "n:", sorted({x["n"] for x in mb})[:6])
