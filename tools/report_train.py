"""Summarise training runs: last epoch vs best epoch (best is chosen on the held-out set, so it is
optimistic; the last epoch is the honest number)."""
import json, sys
from pathlib import Path

def fmt(d):
    return " ".join(f"{k}={v:.3f}" for k, v in d.items())

for tag in sys.argv[1:]:
    p = Path(f"results/train_{tag}.json")
    if not p.exists():
        print(f"{tag}: missing"); continue
    h = json.loads(p.read_text())["history"]
    last = h[-1]
    best = max(h, key=lambda e: e["holdout"].get("30s", 0))
    print(f"{tag:11} epochs={len(h):2d}")
    print(f"   last (ep{last['epoch']:2d}): {fmt(last['holdout'])}")
    print(f"   best (ep{best['epoch']:2d}): {fmt(best['holdout'])}   <- selected on holdout, optimistic")
    if "music_zero_shot" in last:
        ms = [e["music_zero_shot"].get("30s", float("nan")) for e in h]
        print(f"   music zero-shot 30s: first {ms[0]:.3f}  last {ms[-1]:.3f}  min {min(ms):.3f}  max {max(ms):.3f}")
