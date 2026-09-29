"""One statistics pass over every result the project has produced.

Individual experiments each printed their own numbers, sometimes with a test and sometimes without.
A reviewer reads the table as a whole, so the whole table needs one treatment:

  * a distribution-free test per row. Accuracies are paired to a reference (chance, or the same
    listeners under another condition), so the null is sign-flipping the per-listener differences.
    With n listeners there are 2^n sign patterns; up to EXACT_MAX they are all enumerated and the
    p-value is exact, above that 100k are sampled.
  * a bootstrap confidence interval over listeners, which is what the effect size needs.
  * Benjamini-Hochberg correction across every row reported, because the table asks many questions
    of the same data. Rows are declared, not discovered, so the family is fixed in advance.

The input is the result JSONs; nothing is refitted, so this is seconds of work and can be rerun
whenever a result changes.

    python -m eg606.eval.stats --results results --out results/stats.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
EXACT_MAX = 24            # listeners: 2^24 sign patterns, enumerated in chunks
SAMPLES = 100_000
RNG = np.random.default_rng(0)


def sign_flip_p(d: np.ndarray, alternative: str = "greater") -> tuple[float, bool]:
    """Paired permutation test on differences d. Returns (p, exact)."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0 or np.allclose(d, 0):
        return 1.0, True
    obs = d.mean()
    if n <= EXACT_MAX:
        null, exact = _exact_null(d), True
    else:
        null, exact = (RNG.choice((1.0, -1.0), size=(SAMPLES, n)) * d).mean(1), False
    if alternative == "greater":
        p = float((null >= obs).mean())
    elif alternative == "less":
        p = float((null <= obs).mean())
    else:
        p = float((np.abs(null) >= abs(obs)).mean())
    # a permutation p is never 0: the observed arrangement is one of the arrangements
    return max(p, 1.0 / len(null)), exact


def _exact_null(d: np.ndarray, chunk: int = 1 << 16) -> np.ndarray:
    """Means under all 2^n sign flips, built a chunk at a time so memory stays flat."""
    n = len(d)
    bits = 1 << np.arange(n)
    out = np.empty(1 << n)
    for lo in range(0, 1 << n, chunk):
        idx = np.arange(lo, min(lo + chunk, 1 << n))[:, None]
        signs = np.where(idx & bits, -1.0, 1.0)
        out[lo:lo + len(idx)] = (signs * d).mean(1)
    return out


def label_perm_p(a: np.ndarray, b: np.ndarray, n: int = 100_000) -> float:
    """Two independent groups: permute group membership and compare medians.

    A one-sample test of group a against group b's median would treat that median as known and
    ignore b's own spread, which overstates the evidence.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    pool = np.concatenate([a, b])
    obs = abs(np.median(a) - np.median(b))
    hits = 0
    for _ in range(n):
        q = RNG.permutation(pool)
        hits += abs(np.median(q[: len(a)]) - np.median(q[len(a):])) >= obs
    return max((hits + 1) / (n + 1), 1 / (n + 1))


def boot_ci(x: np.ndarray, n: int = 10_000) -> list[float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return [float("nan"), float("nan")]
    means = np.array([x[RNG.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def bh(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg q-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out.tolist()


class Table:
    """Rows accumulate, then get tested and corrected together."""

    def __init__(self):
        self.rows: list[dict] = []

    def add_groups(self, family: str, name: str, a, b, note=""):
        """Two independent groups, compared by median with a label-permutation test."""
        a = np.asarray([x for x in a], float)
        b = np.asarray([x for x in b], float)
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if not len(a) or not len(b):
            return
        rng = np.random.default_rng(0)
        boot = [np.median(a[rng.integers(0, len(a), len(a))])
                - np.median(b[rng.integers(0, len(b), len(b))]) for _ in range(10_000)]
        self.rows.append({"family": family, "row": name, "n": int(len(a)),
                          "mean": float(np.median(a)), "baseline": float(np.median(b)),
                          "diff": float(np.median(a) - np.median(b)),
                          "ci95_diff": [float(np.percentile(boot, 2.5)),
                                        float(np.percentile(boot, 97.5))],
                          "above": int((a > np.median(b)).sum()),
                          "p": label_perm_p(a, b), "exact": False,
                          "alternative": "two-sided", "note": note})

    def add(self, family: str, name: str, values, baseline, alternative="greater", note=""):
        v = np.asarray([x for x in values], dtype=float)
        b = (np.full(len(v), float(baseline)) if np.isscalar(baseline)
             else np.asarray(baseline, dtype=float))
        keep = ~(np.isnan(v) | np.isnan(b))
        v, b = v[keep], b[keep]
        if len(v) == 0:
            return
        d = v - b
        p, exact = sign_flip_p(d, alternative)
        self.rows.append({"family": family, "row": name, "n": int(len(v)),
                          "mean": float(v.mean()), "baseline": float(b.mean()),
                          "diff": float(d.mean()), "ci95_diff": boot_ci(d),
                          "above": int((d > 0).sum()), "p": p, "exact": exact,
                          "alternative": alternative, "note": note})

    def finish(self) -> list[dict]:
        """Correct only the rows that actually state a hypothesis.

        Some rows are descriptive -- "which shift did the sweep select" is a reported quantity, not
        a test -- and carry no p-value. Including them would both inflate the family size and, with
        a NaN in the sort, corrupt every other row's q.
        """
        tested = [r for r in self.rows if np.isfinite(r.get("p", float("nan")))]
        for row, q in zip(tested, bh([r["p"] for r in tested])):
            row["q"] = q
        for row in self.rows:
            row.setdefault("q", float("nan"))
        return self.rows


def _load(root: Path, *names) -> dict | None:
    for n in names:
        hits = sorted(root.rglob(n))
        if hits:
            return json.loads(hits[-1].read_text())
    return None


def collect(root: Path) -> Table:
    t = Table()

    # --- the leakage figure: every protocol against its own chance level
    # leakage.json covers MUSIN-G too, so the older single-dataset file is only a fallback
    multi = _load(root, "leakage.json")
    sources = ([(multi, "multi-dataset")] if multi
               else [(_load(root, "songid_basic.json"), "MUSIN-G")])
    for src, label in sources:
        if not src:
            continue
        datasets = src if label == "multi-dataset" else {"musin_g": {
            "chance_per_listener": {s: 1 / 12 for s in next(iter(src.values()))},
            "conditions": src}}
        for ds, r in datasets.items():
            ch = r["chance_per_listener"]
            for cond, accs in r["conditions"].items():
                subs = sorted(accs)
                t.add("leakage", f"{ds}: {cond} vs chance", [accs[s] for s in subs],
                      [ch[s] for s in subs], note=f"chance {np.mean(list(ch.values())):.3f}")
            # the comparison that matters: what the naive protocol adds over a held-out listener
            c = r["conditions"]
            for a, b in (("within_naive", "loso"), ("within_naive", "cross_repetition"),
                         ("within_naive", "silence_naive")):
                if a in c and b in c:
                    subs = sorted(set(c[a]) & set(c[b]))
                    if subs:
                        t.add("leakage", f"{ds}: {a} minus {b}", [c[a][s] for s in subs],
                              [c[b][s] for s in subs], note="paired over listeners")

    # --- response latency: is MUSIN-G's onset response later than an independent dataset's?
    erp = _load(root, "onset_erp_sharp.json", "onset_erp_all.json")
    if erp and "musin_g" in erp and "nmed_t" in erp:
        a = [v["gfp_peak_ms"] for v in erp["musin_g"]["per_listener"].values()]
        b = [v["gfp_peak_ms"] for v in erp["nmed_t"]["per_listener"].values()]
        # independent groups, so each is compared against the other's median
        t.add_groups("latency", "onset GFP peak, MUSIN-G minus NMED-T", a, b, note="ms")
    # every band, because a fixed hardware delay should be the same in all of them
    for path in sorted(root.rglob("trf_compare_b*.json")):
        trf = json.loads(path.read_text())
        ref = trf.get("datasets", {}).get("speech (SparrKULee)")
        if not ref:
            continue
        band = trf.get("band") or [0.5, 32.0]
        tag = f"{band[0]:g}-{band[1]:g} Hz"
        for name, r in trf["datasets"].items():
            if name == "speech (SparrKULee)":
                continue
            short = name.split("(")[-1].rstrip(")")
            if "gfp_shift_ci95_ms" not in r:
                continue
            lo, hi = r["gfp_shift_ci95_ms"]
            t.rows.append({"family": "latency",
                           "row": f"alignment shift, {short} vs speech, {tag}",
                           "n": r["listeners"], "mean": r["gfp_shift_vs_speech_ms"],
                           "baseline": 0.0, "diff": r["gfp_shift_vs_speech_ms"],
                           "ci95_diff": [lo, hi], "above": 0,
                           "p": float(r.get("gfp_shift_p", float("nan"))),
                           "exact": False, "alternative": "two-sided",
                           "note": "ms; listener bootstrap"})
            if band == [4.0, 8.0]:
                t.add_groups("latency", f"TRF peak, {short} minus speech, {tag}",
                             r["peaks_ms"], ref["peaks_ms"], note="ms")

    # --- OpenMIIR: the speech decoder applied with no montage interpolation at all
    for path in sorted(root.rglob("openmiir/latency_*.json")):
        d = json.loads(path.read_text())
        band = d.get("band") or [0.5, 32.0]
        tag = f"{band[0]:g}-{band[1]:g} Hz"
        sweep = d.get("sweep", [])
        if not sweep:
            continue
        # window keys look like "3s"; "shift_ms" also ends in "s", so match the shape properly
        wins = sorted((k for k in sweep[0] if re.fullmatch(r"\d+(\.\d+)?s", k)),
                      key=lambda k: float(k[:-1]))
        if not wins:
            continue
        win = wins[-1]
        best = max(sweep, key=lambda r: r.get(win, 0))
        at_zero = next((r for r in sweep if r["shift_ms"] == 0), None)
        t.rows.append({"family": "latency",
                       "row": f"shift chosen by the sweep, OpenMIIR, {tag}",
                       "n": 0, "mean": best["shift_ms"], "baseline": 0.0,
                       "diff": best["shift_ms"], "ci95_diff": [float("nan")] * 2,
                       "above": 0, "p": float("nan"), "exact": False,
                       "alternative": "two-sided",
                       "note": f"ms; {win} acc {best.get(win, float('nan')):.3f}, "
                               f"unshifted {at_zero.get(win, float('nan')):.3f}"
                               if at_zero else "ms"})

    # --- match-mismatch decoding: each window against the 0.5 chance of a two-way choice
    for name, label in (("protocol_none_onset.json", "linear, onset"),
                        ("protocol_b0408.json", "linear, 4-8 Hz")):
        src = _load(root, name)
        if not src:
            continue
        for split, per in src.get("splits", {}).items():
            if not isinstance(per, dict):
                continue
            for win, vals in _per_window(per).items():
                t.add("decoding", f"{label}: {split} at {win}", vals, 0.5)

    # --- the deep model, per listener, seeds averaged
    for tag, label in (("cv_music_lcv", "MUSIN-G, new listener"),
                       ("cv_music_songcv", "MUSIN-G, new song"),
                       ("cv_bach_lcv", "Bach, new listener")):
        scratch = _seeds(root, f"{tag}_s*.json")
        tuned = _seeds(root, f"{tag}_ft_s*.json")
        for label2, d in (("from scratch", scratch), ("speech-pretrained", tuned)):
            for win, vals in d.items():
                t.add("decoding", f"{label} ({label2}) at {win}", list(vals.values()), 0.5)
        # transfer: the same listeners, with and without speech pretraining
        for win in sorted(set(scratch) & set(tuned)):
            subs = sorted(set(scratch[win]) & set(tuned[win]))
            if subs:
                t.add("transfer", f"{label}: speech pretraining at {win}",
                      [tuned[win][s] for s in subs], [scratch[win][s] for s in subs],
                      note="paired over listeners")
    return t


def _per_window(per_listener: dict) -> dict:
    """{window: [per-listener accuracy]} from a {listener: {window: acc}} mapping."""
    out: dict[str, list] = {}
    for sub, v in per_listener.items():
        if not isinstance(v, dict):
            continue
        for win, acc in (v.get("mm", v)).items():
            if isinstance(acc, (int, float)):
                out.setdefault(win, []).append(acc)
    return out


def _seeds(root: Path, pattern: str) -> dict:
    """{window: {listener: mean accuracy over seeds}} for one configuration."""
    acc: dict[str, dict[str, list]] = {}
    for path in sorted(root.rglob(pattern)):
        d = json.loads(path.read_text()).get("per_listener", {})
        for sub, wins in d.items():
            for win, v in wins.items():
                acc.setdefault(win, {}).setdefault(sub, []).append(v)
    return {w: {s: float(np.mean(v)) for s, v in subs.items()} for w, subs in acc.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.eval.stats")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rows = collect(Path(args.results)).finish()
    order = {"leakage": 0, "latency": 1, "decoding": 2, "transfer": 3}
    rows.sort(key=lambda r: order.get(r["family"], 9))
    if not rows:
        print("no results found")
        return 1

    print(f"\n=== every reported number, one test each, Benjamini-Hochberg over all {len(rows)} rows")
    print("p: paired sign-flip permutation test (exact where marked *)   q: BH-corrected")
    fam = None
    for r in rows:
        if r["family"] != fam:
            fam = r["family"]
            print(f"\n[{fam}]")
            print(f"  {'row':<52}{'n':>3}{'value':>8}{'base':>8}{'diff':>8}"
                  f"{'95% CI':>18}{'p':>10}{'q':>10}")
        ci = r["ci95_diff"]
        star = "*" if r["exact"] else " "
        ci_s = ("        descriptive" if not np.isfinite(ci[0])
                else f"  [{ci[0]:+6.3f},{ci[1]:+6.3f}]")
        p_s = "        -" if not np.isfinite(r["p"]) else f"{r['p']:>9.2g}"
        q_s = "        -" if not np.isfinite(r["q"]) else f"{r['q']:>9.2g}"
        print(f"  {r['row']:<52}{r['n']:>3}{r['mean']:>8.3f}{r['baseline']:>8.3f}"
              f"{r['diff']:>+8.3f}{ci_s}{p_s}{star}{q_s}")
    tested = [r for r in rows if np.isfinite(r["q"])]
    sig = sum(r["q"] < 0.05 for r in tested)
    print(f"\n{sig} of {len(tested)} tested rows survive correction at q < 0.05 "
          f"({len(rows) - len(tested)} descriptive rows not corrected)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
