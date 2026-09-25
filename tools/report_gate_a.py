"""Statistics for a Gate A run: is the across-listener accuracy above chance?"""
import json, sys
import numpy as np
from scipy import stats

path = sys.argv[1] if len(sys.argv) > 1 else "results/gate_a.json"
res = json.loads(open(path).read())
subs = sorted(res)
print(f"{path}: {len(subs)} listeners\n")

r = np.array([res[s]["r_mean"] for s in subs])
print(f"envelope reconstruction r: mean {r.mean():.4f}  sd {r.std(ddof=1):.4f}  "
      f"range [{r.min():.4f}, {r.max():.4f}]")
t, p = stats.ttest_1samp(r, 0.0)
print(f"  vs 0: t={t:.2f} p={p:.4g}\n")

for win in sorted(res[subs[0]]["mm"], key=lambda k: float(k[:-1])):
    a = np.array([res[s]["mm"][win] for s in subs])
    above = int((a > 0.5).sum())
    t, p = stats.ttest_1samp(a, 0.5)
    try:
        w, pw = stats.wilcoxon(a - 0.5)
    except ValueError:
        w, pw = float("nan"), float("nan")
    pb = stats.binomtest(above, len(a), 0.5).pvalue
    sil = np.array([res[s]["silence_mm"][win] for s in subs], dtype=float)
    sil_txt = "n/a" if np.isnan(sil).all() else f"{np.nanmean(sil):.3f}"
    print(f"match-mismatch {win:>4}: mean {a.mean():.3f} sd {a.std(ddof=1):.3f} | "
          f"{above}/{len(a)} above chance")
    print(f"    t-test vs 0.5: t={t:.2f} p={p:.4g} | Wilcoxon p={pw:.4g} | sign test p={pb:.4g}"
          f" | silence control {sil_txt}")

print("\nper listener (5s / 10s / r):")
for s in subs:
    m = res[s]["mm"]
    keys = sorted(m, key=lambda k: float(k[:-1]))
    print(f"  {s}: " + "  ".join(f"{k}={m[k]:.3f}" for k in keys) + f"  r={res[s]['r_mean']:+.4f}")
