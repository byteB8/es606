"""One report for the single objective: which song is a new listener hearing?

Reads results/newlistener/*.json (eval/newlistener.py and eval/newlistener_baselines.py) and
writes results/newlistener/report.md and report.json.

Declared before any full result existed (2026-11-04), so that no window or comparison is chosen
after seeing the numbers:

  * Primary window: 10 s (about 10 decisions per song per listener, 2,400 in all). Every other
    window, from 2 s to the whole 100 s, is reported beside it as a curve, never alone.
  * Unit of analysis: the listener (n = 20). Tests are sign-flip permutations of per-listener
    differences; Benjamini-Hochberg correction over every row of the comparison table below.
  * Main comparisons (new listener, onset target, 10 s):
      decoder vs chance; decoder vs its time-shift control; decoder vs logistic regression;
      decoder vs CNN; decoder on new listener AND new song vs chance.
  * Frequency-content comparisons (same decoder, same 10 s window):
      audio 0-4 kHz vs 0-0.5 kHz (does sound above 0.5 kHz help?);
      audio 0-4 kHz vs 0-2 kHz (does the top octave help?);
      8 frequency bands vs one averaged curve (does keeping the spectral detail help?);
      EEG at 128 Hz vs 64 Hz, both 0.5-32 Hz (does the sampling rate matter?);
      EEG 0.5-45 Hz vs 0.5-32 Hz, both at 128 Hz (does EEG above 32 Hz help?);
      32-45 Hz gamma envelope alone vs chance (is there music information up there at all?).

    python tools/report_newlistener.py [--dir results/newlistener] [--window 10]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eg606.eval.stats import bh, boot_ci, sign_flip_p  # noqa: E402

CHANCE = 1 / 12


def acc(per_listener: dict, window: str, cond: str = "real") -> dict[str, float]:
    out = {}
    for sub, v in per_listener.items():
        h, n = v["win"][window][cond]
        if n:
            out[sub] = h / n
    return out


def paired(a: dict, b: dict | float):
    """Mean difference a - b over shared listeners, sign-flip p, bootstrap CI, n."""
    if isinstance(b, float):
        d = np.array([a[s] - b for s in sorted(a)])
    else:
        subs = sorted(set(a) & set(b))
        d = np.array([a[s] - b[s] for s in subs])
    return {"diff": float(d.mean()), "ci": boot_ci(d), "p": sign_flip_p(d)[0], "n": len(d),
            "positive": int((d > 0).sum())}


def load(d: Path, name: str):
    p = d / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/newlistener")
    ap.add_argument("--window", default="10")
    ap.add_argument("--decoder", default="decoder_basic_lam11",
                    help="main decoder result (the 1e3..1e11 penalty grid; see the notebook)")
    ap.add_argument("--suffix", default="_lam11", help="suffix of the 128 Hz / gamma results")
    args = ap.parse_args(argv)
    d, w = Path(args.dir), args.window

    dec = load(d, args.decoder)
    if dec is None:
        print(f"no {args.decoder}.json in {d}")
        return 1
    logreg, cnn = load(d, "logreg_basic"), load(d, "cnn_basic")
    eeg_arms = {k: load(d, k + args.suffix)
                for k in ("decoder_fs128lp32", "decoder_fs128lp45", "decoder_gamma")}
    loso = dec["splits"]["loso"]
    pl = loso["per_listener"]
    windows = list(dec["windows"])
    lines = ["# Which song is a new listener hearing? (MUSIN-G, 20 listeners, 12 songs)", "",
             f"Chance 1/12 = 0.083. Decisions use the first {dec['n_common_s']:.1f} s of each trial. "
             f"Primary window {w} s (declared in advance).", ""]

    # ---- accuracy curves
    def curve_row(name, per_listener, cond="real"):
        cells = []
        for x in windows:
            a = acc(per_listener, x, cond) if x in next(iter(per_listener.values()))["win"] else {}
            cells.append(f"{np.mean(list(a.values())):.3f}" if a else "n/a")
        return f"| {name} | " + " | ".join(cells) + " |"

    lines += ["## Accuracy by decision window (mean over listeners)", "",
              "| method | " + " | ".join(x + (" s" if x != "all" else " (100 s)") for x in windows) + " |",
              "|---|" + "---|" * len(windows)]
    lines.append(curve_row("**decoder, new listener**", pl["onset"]))
    lines.append(curve_row("control: EEG shifted in time", pl["onset"], "shift"))
    if "loso_song" in dec["splits"]:
        lines.append(curve_row("decoder, new listener and new song",
                               dec["splits"]["loso_song"]["per_listener"]["onset"]))
    for name, res in (("logistic regression (band power)", logreg), ("CNN (spectrograms)", cnn)):
        if res:
            lines.append(curve_row(name, res["per_listener"]))
    dec_n = loso["summary"]["onset"]
    lines.append("| decisions | " + " | ".join(str(dec_n[x]["real"]["decisions"]) for x in windows) + " |")
    lines.append("")

    # ---- declared comparisons
    rows = []
    a_dec = acc(pl["onset"], w)
    rows.append(("decoder vs chance", paired(a_dec, CHANCE)))
    rows.append(("decoder vs time-shift control", paired(a_dec, acc(pl["onset"], w, "shift"))))
    if logreg:
        rows.append(("decoder vs logistic regression", paired(a_dec, acc(logreg["per_listener"], w))))
    if cnn:
        rows.append(("decoder vs CNN", paired(a_dec, acc(cnn["per_listener"], w))))
    if "loso_song" in dec["splits"]:
        rows.append(("new listener and new song vs chance",
                     paired(acc(dec["splits"]["loso_song"]["per_listener"]["onset"], w), CHANCE)))
    hf = [("audio 0-4 kHz vs 0-0.5 kHz", "onset", "onset_lt500"),
          ("audio 0-4 kHz vs 0-2 kHz", "onset", "onset_lt2k"),
          ("8 bands vs one averaged curve", "onset_8band", "onset")]
    for name, a, b in hf:
        if a in pl and b in pl:
            rows.append((name, paired(acc(pl[a], w), acc(pl[b], w))))
    arm = {k: (v["splits"]["loso"]["per_listener"]["onset"] if v else None)
           for k, v in eeg_arms.items()}
    if arm["decoder_fs128lp32"]:
        rows.append(("EEG 128 Hz vs 64 Hz (both 0.5-32 Hz)",
                     paired(acc(arm["decoder_fs128lp32"], w), a_dec)))
    if arm["decoder_fs128lp32"] and arm["decoder_fs128lp45"]:
        rows.append(("EEG 0.5-45 Hz vs 0.5-32 Hz (both 128 Hz)",
                     paired(acc(arm["decoder_fs128lp45"], w), acc(arm["decoder_fs128lp32"], w))))
    if arm["decoder_gamma"]:
        rows.append(("32-45 Hz gamma envelope vs chance", paired(acc(arm["decoder_gamma"], w), CHANCE)))

    qs = bh([r[1]["p"] for r in rows])
    lines += [f"## Declared comparisons at {w} s (listener-level, BH over all rows)", "",
              "| comparison | difference | 95% CI | listeners > 0 | p | q |", "|---|---|---|---|---|---|"]
    for (name, r), q in zip(rows, qs):
        lines.append(f"| {name} | {r['diff']:+.3f} | [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | "
                     f"{r['positive']}/{r['n']} | {r['p']:.2g} | {q:.2g} |")
    lines.append("")

    # ---- frequency content, all windows
    lines += ["## How much of the music's frequency content to follow (decoder, new listener)", "",
              "| music target | " + " | ".join(x + (" s" if x != "all" else "") for x in windows) + " |",
              "|---|" + "---|" * len(windows)]
    names = {"onset_lt500": "onsets, audio 0-0.5 kHz", "onset_lt1k": "onsets, audio 0-1 kHz",
             "onset_lt2k": "onsets, audio 0-2 kHz", "onset": "onsets, audio 0-4 kHz (all)",
             "onset_8band": "onsets in 8 bands, 0-4 kHz", "env": "loudness envelope"}
    for k, label in names.items():
        if k in pl:
            lines.append(curve_row(label, pl[k]))
    lines += ["", "## EEG sampling rate and frequency range (decoder, new listener, onsets 0-4 kHz)", "",
              "| EEG | " + " | ".join(x + (" s" if x != "all" else "") for x in windows) + " |",
              "|---|" + "---|" * len(windows)]
    lines.append(curve_row("64 Hz, 0.5-32 Hz (main)", pl["onset"]))
    for k, label in (("decoder_fs128lp32", "128 Hz, 0.5-32 Hz"), ("decoder_fs128lp45", "128 Hz, 0.5-45 Hz"),
                     ("decoder_gamma", "32-45 Hz amplitude envelope only")):
        if arm[k]:
            lines.append(curve_row(label, arm[k]))
    lines.append("")

    text = "\n".join(lines)
    (d / "report.md").write_text(text)
    (d / "report.json").write_text(json.dumps(
        {"window": w, "comparisons": [{"name": n, **r, "q": q} for (n, r), q in zip(rows, qs)]},
        indent=1))
    print(text)
    print(f"wrote {d / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
