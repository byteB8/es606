"""The figures the paper needs, drawn from the result JSONs.

Kept separate from the analyses so a figure can be redrawn without recomputing anything, and so the
numbers in a figure are provably the numbers in the table.

    python -m eg606.eval.figs --results results --out results/figs
"""
from __future__ import annotations

import argparse
import json
import logging
import re
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
# the long labels collide once the figure is drawn at print width
COND_SHORT = {"within_naive": "within\nrecording", "pooled_naive": "pooled",
              "cross_repetition": "other\nblock", "loso": "held-out\nlistener",
              "loso_song": "held-out\nlistener", "silence_naive": "SILENCE",
              "silence_loso": "silence,\nheld-out", "within_naive_shuffled": "shuffled\nlabels"}


# IEEEtran at 10pt: a column is 3.5 in and the full text block 7.16 in. A figure drawn at screen
# size and then scaled into a column shrinks its text to illegibility, so the paper variants are
# drawn at final size with type that is already the right number of points.
PAPER = {"column": 3.4, "full": 7.0}
PAPER_RC = {"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
            "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
            "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01, "axes.linewidth": 0.6,
            "xtick.major.width": 0.6, "ytick.major.width": 0.6}
STYLE = {"paper": False}


def _save(fig, out: Path, stem: str) -> None:
    """Write PNG for the README and, in paper mode, a vector PDF for LaTeX."""
    fig.savefig(out / f"{stem}.png", dpi=180)
    if STYLE["paper"]:
        fig.savefig(out / f"{stem}.pdf")
        log.info("wrote %s", out / f"{stem}.pdf")
    log.info("wrote %s", out / f"{stem}.png")


def _figsize(width: str, height: float):
    return (PAPER[width] if STYLE["paper"] else {"column": 7.0, "full": 14.0}[width], height)


def _load(root: Path, name: str):
    hits = sorted(root.rglob(name))
    return json.loads(hits[-1].read_text()) if hits else None


def _mean_mm(per_fold: dict, win: str) -> float:
    vals = [v["mm"][win] for v in per_fold.values() if win in v.get("mm", {})]
    return float(np.nanmean(vals)) if vals else float("nan")


def collect_controls(root: Path) -> list[dict]:
    """Every result that has a matched derangement control, as (label, real, control).

    A match-mismatch score is only interpretable next to the score the same model gets when the
    EEG no longer corresponds to the audio. Collecting both in one place is the point of the
    figure: the gap is the claim, and the control height is the leak.
    """
    rows = []
    prot = _load(root, "protocol_control.json")
    if prot and prot.get("controls"):
        for split, label in (("loso", "linear, held-out listener"),
                             ("song", "linear, held-out song")):
            if split in prot["splits"] and split in prot["controls"]:
                rows.append({"label": label, "window": "30s",
                             "real": _mean_mm(prot["splits"][split], "30s"),
                             "control": _mean_mm(prot["controls"][split], "30s")})
    # both datasets write a file of the same name, so match on the directory too: an rglob that
    # takes the last hit would quietly score MUSIN-G's row from OpenMIIR's file
    for name, label, win in (("bhaskar/latency_broad_ctl.json",
                              "linear zero-shot, speech->MUSIN-G", "30s"),
                             ("openmiir/latency_broad_ctl.json",
                              "linear zero-shot, speech->OpenMIIR", "3s")):
        d = _load(root, name)
        if not d or not d.get("control"):
            continue
        sweep = d.get("sweep", [])
        best = max(sweep, key=lambda r: r.get(win, 0)) if sweep else {}
        real, ctrl = best.get(win, float("nan")), d["control"].get(win, float("nan"))
        if np.isfinite(real) and np.isfinite(ctrl):
            rows.append({"label": label, "window": win, "real": real, "control": ctrl})
    for name, label, win in (("cv_music_control.json", "deep v2, held-out listener", "30s"),
                             ("cv_bach_control.json", "deep v2, Bach", "10s")):
        d = _load(root, name)
        if not d or "controls" not in d:
            continue
        real = float(np.nanmean([v[win] for v in d["per_listener"].values() if win in v]))
        ctl = d["controls"].get("shuffle-eeg", {})
        ctrl = float(np.nanmean([v[win] for v in ctl.values() if win in v])) if ctl else float("nan")
        if np.isfinite(real) and np.isfinite(ctrl):
            rows.append({"label": label, "window": win, "real": real, "control": ctrl})
    return rows


def controls_figure(root: Path, out: Path) -> None:
    """What each result scores, beside what it scores with the EEG destroyed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = collect_controls(root)
    if not rows:
        log.warning("no control results; skipping the controls figure")
        return
    rows.sort(key=lambda r: (r["real"] - r["control"]), reverse=True)
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=_figsize(
        "full", (0.26 * len(rows) + 0.95) if STYLE["paper"] else (0.62 * len(rows) + 2.2)))
    ax.barh(y + 0.19, [r["real"] for r in rows], 0.36, color="#3d6fb4", label="as reported")
    ax.barh(y - 0.19, [r["control"] for r in rows], 0.36, color="#c0392b",
            label="same model, EEG no longer matches the audio")
    ax.axvline(0.5, color="k", ls="--", lw=1)
    for i, r in enumerate(rows):
        ax.text(max(r["real"], r["control"]) + 0.008, i,
                f"genuine {r['real'] - r['control']:+.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['label']}  ({r['window']})" for r in rows], fontsize=8.5)
    ax.set_xlim(0.45, 1.06)
    ax.set_xlabel("match-mismatch accuracy (dashed line: chance)")
    if not STYLE["paper"]:
        ax.set_title("Every match-mismatch result beside its own control\n"
                     "A bar pair that nearly touches is a result the EEG is barely contributing to",
                     fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, out, "controls")


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

    fig, axes = plt.subplots(1, len(names), figsize=_figsize("full", 2.5 if STYLE["paper"] else 5.0),
                             sharey=True)
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
        if STYLE["paper"]:
            ax.set_xticklabels([COND_SHORT.get(c, c) for c in conds], fontsize=5.6,
                               rotation=0, linespacing=0.95)
        else:
            ax.set_xticklabels([COND_LABEL.get(c, c) for c in conds], fontsize=8,
                               rotation=35, ha="right")
        ax.set_title(titles.get(ds, ds), fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("accuracy / chance")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    if not STYLE["paper"]:
        fig.suptitle("The same data and the same classifier, split five ways "
                     "(dashed line: chance)", fontsize=10)
    fig.tight_layout()
    _save(fig, out, "leakage")


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

    fig, ax = plt.subplots(figsize=_figsize("column", 2.3 if STYLE["paper"] else 4))
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
    _save(fig, out, "latency_bands")


def sweep_figure(root: Path, out: Path) -> None:
    """Accuracy against the shift applied to the music, for all three music datasets.

    This is the timing result in one picture: the same speech decoder, the same sweep, applied to
    three datasets. MUSIN-G peaks away from zero; OpenMIIR, whose authors corrected their onsets
    with a recorded audio marker, peaks at zero.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Bach is deliberately absent: its match-mismatch score survives the derangement control, so
    # the shift it selects is not interpretable as a response latency.
    sources = [("MUSIN-G", "bhaskar/latency_broad.json", "#c0392b"),
               ("OpenMIIR", "openmiir/latency_broad.json", "#1e8449")]
    fig, ax = plt.subplots(figsize=_figsize("column", 2.4 if STYLE["paper"] else 4.4))
    found = 0
    for label, pattern, color in sources:
        hits = sorted(root.rglob(pattern))
        if not hits:
            log.warning("no %s for the sweep figure", pattern)
            continue
        d = json.loads(hits[-1].read_text())
        sweep = d.get("sweep", [])
        wins = sorted((k for k in sweep[0] if re.fullmatch(r"\d+(\.\d+)?s", k)),
                      key=lambda k: float(k[:-1]))
        if not wins:
            continue
        win = wins[-1]
        xs = [r["shift_ms"] for r in sweep]
        ys = [r.get(win, float("nan")) for r in sweep]
        ax.plot(xs, ys, "o-", color=color, markersize=3 if STYLE["paper"] else 4,
                linewidth=1.0 if STYLE["paper"] else 1.5,
                label=f"{label} ({win})")
        best = max(sweep, key=lambda r: r.get(win, 0))
        ax.axvline(best["shift_ms"], color=color, ls=":", lw=1)
        found += 1
    if not found:
        plt.close(fig)
        return
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel("shift applied to the music (ms)" if STYLE["paper"]
                  else "shift applied to the music (ms; positive = music treated as later)")
    ax.set_ylabel("match\u2013mismatch accuracy")
    if not STYLE["paper"]:
        ax.set_title("One speech decoder, three music datasets\n"
                     "(dotted lines: the shift each dataset selects)", fontsize=10)
    ax.legend(frameon=False, fontsize=6.5 if STYLE["paper"] else 8,
              loc="lower center", ncol=2 if STYLE["paper"] else 1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out, "shift_sweeps")


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
    axes[0].set_ylabel("match\u2013mismatch accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Speech pretraining as music data is withheld: the gain does not grow "
                 "when data is scarce", fontsize=10)
    fig.tight_layout()
    _save(fig, out, "scaling")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.eval.figs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figs")
    ap.add_argument("--paper", action="store_true",
                    help="draw at final column width with paper-sized type, and write PDFs")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.paper:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        STYLE["paper"] = True
        plt.rcParams.update(PAPER_RC)
    leakage_figure(root, out)
    controls_figure(root, out)
    latency_figure(root, out)
    sweep_figure(root, out)
    scaling_figure(root, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
