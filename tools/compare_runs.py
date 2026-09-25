"""Paired comparison of two cross-validated runs (same listeners in both arms)."""
import json, sys
import numpy as np
from pathlib import Path
from scipy import stats


def load(path):
    d = json.loads(Path(path).read_text())
    return d.get("per_listener", d.get("speech_test", {}))


a_path, b_path = sys.argv[1], sys.argv[2]
A, B = load(a_path), load(b_path)
shared = sorted(set(A) & set(B))
print(f"A = {Path(a_path).stem}\nB = {Path(b_path).stem}\n{len(shared)} shared listeners\n")
print(f"{'window':>8}{'A':>9}{'B':>9}{'B-A':>9}{'wins':>7}   paired t      Wilcoxon")
for w in sorted(A[shared[0]], key=lambda k: float(k[:-1])):
    a = np.array([A[s][w] for s in shared])
    b = np.array([B[s][w] for s in shared])
    d = b - a
    t, p = stats.ttest_rel(b, a)
    try:
        _, pw = stats.wilcoxon(d)
    except ValueError:
        pw = float("nan")
    print(f"{w:>8}{a.mean():9.3f}{b.mean():9.3f}{d.mean():+9.3f}{(d > 0).sum():4d}/{len(d)}"
          f"   t={t:+5.2f} p={p:7.4f}   p={pw:.4f}")
