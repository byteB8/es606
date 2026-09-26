"""The figures the paper needs, drawn from the result JSONs.

Kept separate from the analyses so a figure can be redrawn without recomputing anything, and so the
numbers in a figure are provably the numbers in the table.

    python -m eg606.eval.figs --results results --out results/figs
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

COND_LABEL = {
    "within_naive": "within recording\n(published protocol)",
    "pooled_naive": "pooled, random windows",
    "cross_repetition": "same stimulus, other block",
    "loso": "held-out listener",
    "loso_song": "held-out listener",
    "silence_naive": "SILENCE before the song",
    "silence_loso": "silence, held-out listener",
    "within_naive_shuffled": "shuffled labels",
}
ORDER = ["within_naive", "pooled_naive", "silence_naive", "cross_repetition",
         "loso", "loso_song", "silence_loso", "within_naive_shuffled"]


def _load(root: Path, name: str):
    hits = sorted(root.rglob(name))
    return json.loads(hits[-1].read_text()) if hits else None


def leakage_figure(root: Path, out: Path) -> None:
    """Accuracy as a multiple of chance, per protocol, per dataset, linear vs CNN."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lin = _load(root, "leakage.json")
    if not lin:
        log.warning("no leakage.json; skipping the leakage figure")
        return
    cnn = {ds: _load(root, f"cnn_{ds}.json") for ds in lin}
    names = [d for d in ("musin_g", "nmed_t", "nmed_h") if d in lin]
    titles = {"musin_g": "MUSIN-G (12 songs)", "nmed_t": "NMED-T (10 songs)",
              "nmed_h": "NMED-H (4 stimuli/listener)"}

    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 5.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, names):
        r = lin[ds]
        ch = r["chance_per_listener"]
        conds = [c for c in ORDER if c in r["conditions"]]
        x = np.arange(len(conds))
        for off, (src, label, color) in enumerate(
                ((r["conditions"], "linear (band power)", "#3d6fb4"),
                 ((cnn[ds] or {}).get("conditions", {}), "CNN (spectrogram)", "#c0392b"))):
            vals, errs, xs = [], [], []
            for i, c in enumerate(conds):
                accs = src.get(c)
                if not accs:
                    continue
                rel = np.array([accs[s] / ch.get(s, r["chance_pooled"]) for s in accs])
                vals.append(rel.mean())
                errs.append(rel.std() / np.sqrt(len(rel)))
                xs.append(i + (off - 0.5) * 0.38)
            if vals:
                ax.bar(xs, vals, 0.36, yerr=errs, capsize=2, color=color, label=label)
        ax.axhline(1, color="k", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABEL.get(c, c) for c in conds], fontsize=8,
                           rotation=35, ha="right")
        ax.set_title(titles.get(ds, ds), fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("accuracy / chance")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("The same data and the same classifier, split five ways "
                 "(dashed line: chance)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "leakage.png", dpi=180)
    log.info("wrote %s", out / "leakage.png")


def latency_figure(root: Path, out: Path) -> None:
    """The alignment shift in every band: a fixed delay does not care about frequency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(root.rglob("trf_compare_b*.json"))
    if not files:
        log.warning("no trf_compare results; skipping the latency figure")
        return
    rows = []
    for f in files:
        d = json.loads(f.read_text())
        band = tuple(d["band"]) if d.get("band") else (0.5, 32.0)
        for name, r in d["datasets"].items():
            # an earlier run of the same band wrote the shift without a bootstrap; a point with no
            # interval would be drawn as a bare dot and read as a precise measurement
            if "gfp_shift_ci95_ms" in r:
                rows.append((band, name, r["gfp_shift_vs_speech_ms"], r["gfp_shift_ci95_ms"]))
    if not rows:
        return
    bands = sorted({b for b, *_ in rows}, key=lambda b: b[1] - b[0])
    datasets = sorted({n for _, n, *_ in rows})
    colors = {"music (MUSIN-G)": "#c0392b", "music (Bach)": "#2c6fbb"}

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for j, name in enumerate(datasets):
        xs, ys, lo, hi = [], [], [], []
        for i, b in enumerate(bands):
            m = [r for r in rows if r[0] == b and r[1] == name]
            if not m:
                continue
            xs.append(i + (j - (len(datasets) - 1) / 2) * 0.18)
            ys.append(m[0][2])
            lo.append(m[0][2] - m[0][3][0])
            hi.append(m[0][3][1] - m[0][2])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", capsize=3, color=colors.get(name),
                    label=name, markersize=5)
    ax.axhline(0, color="k", lw=0.9)
    ax.axhline(62.5, color="#c0392b", ls=":", lw=1)
    ax.text(len(bands) - 0.4, 64, "+62 ms", color="#c0392b", fontsize=8, va="bottom", ha="right")
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([f"{a:g}-{b:g}" for a, b in bands])
    ax.set_xlabel("EEG band (Hz)")
    ax.set_ylabel("shift aligning music onto speech (ms)")
    ax.set_title("Shift aligning the music response onto speech, by band\n"
                 "(MUSIN-G lands on +62 ms in every band; Bach does not settle)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "latency_bands.png", dpi=180)
    log.info("wrote %s", out / "latency_bands.png")


def scaling_figure(root: Path, out: Path) -> None:
    """How much music EEG is speech pretraining worth?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms: dict[str, dict[float, dict]] = {"scratch": {}, "ft": {}}
    for f in sorted(root.rglob("cv_scale_music_*.json")):
        d = json.loads(f.read_text())
        arm = "ft" if "_ft_" in f.name else "scratch"
        arms[arm][d["args"]["train_fraction"]] = d["per_listener"]
    if not arms["scratch"]:
        log.warning("no scaling results yet; skipping the scaling figure")
        return
    windows = sorted({w for a in arms.values() for p in a.values()
                      for v in p.values() for w in v}, key=lambda s: float(s[:-1]))
    fig, axes = plt.subplots(1, len(windows), figsize=(4 * len(windows), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, win in zip(axes, windows):
        for arm, color, label in (("scratch", "#7f8c8d", "music only"),
                                  ("ft", "#c0392b", "speech pretraining")):
            fr = sorted(arms[arm])
            if not fr:
                continue
            m = [np.mean([v[win] for v in arms[arm][f].values() if win in v]) for f in fr]
            se = [np.std([v[win] for v in arms[arm][f].values() if win in v])
                  / np.sqrt(max(1, len(arms[arm][f]))) for f in fr]
            ax.errorbar(fr, m, yerr=se, fmt="o-", color=color, label=label, capsize=3)
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_xscale("log")
        ticks = sorted({f for a in arms.values() for f in a})
        ax.set_xticks(ticks)
        ax.set_xticklabels([("all" if t >= 1 else f"1/{round(1 / t)}") for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("music training data kept")
        ax.set_title(f"{win} windows", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("match-mismatch accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Speech pretraining as music data is withheld: the gain does not grow "
                 "when data is scarce", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "scaling.png", dpi=180)
    log.info("wrote %s", out / "scaling.png")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.eval.figs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figs")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    leakage_figure(root, out)
    latency_figure(root, out)
    scaling_figure(root, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
