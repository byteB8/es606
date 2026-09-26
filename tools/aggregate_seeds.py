"""Average per-listener results over seeds, then compare two arms with a paired test.

    python tools/aggregate_seeds.py "results/*/cv_music_songcv_s*.json" "results/*/cv_music_songcv_ft_s*.json"
"""
import glob, json, sys
from collections import defaultdict
import numpy as np
from scipy import stats


def merge(pattern):
    runs = sorted(glob.glob(pattern))
    per = defaultdict(lambda: defaultdict(list))
    for f in runs:
        d = json.loads(open(f).read()).get("per_listener", {})
        for sub, row in d.items():
            for w, v in row.items():
                per[sub][w].append(v)
    return runs, {s: {w: float(np.mean(v)) for w, v in r.items()} for s, r in per.items()}


runs_a, A = merge(sys.argv[1])
runs_b, B = merge(sys.argv[2])
print(f"A: {len(runs_a)} run(s)  B: {len(runs_b)} run(s)")
shared = sorted(set(A) & set(B))
print(f"{len(shared)} listeners\n{'window':>8}{'A':>9}{'B':>9}{'B-A':>9}{'wins':>8}   paired test")
for w in sorted(A[shared[0]], key=lambda k: float(k[:-1])):
    a = np.array([A[s][w] for s in shared])
    b = np.array([B[s][w] for s in shared])
    t, p = stats.ttest_rel(b, a)
    print(f"{w:>8}{a.mean():9.3f}{b.mean():9.3f}{(b-a).mean():+9.3f}{(b>a).sum():5d}/{len(a)}"
          f"   t={t:+5.2f} p={p:.4f}")
