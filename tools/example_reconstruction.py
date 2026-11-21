"""Real decoder output for the slides: one held-out listener, all 12 songs.

Refits the new-listener decoder for a single fold exactly as eval/newlistener.py does (onset target,
basic cleaning, the penalty that fold chose in the given result file) and saves what the slides
show: each song's reconstruction, the 12 candidate onset curves, the 10 s window scores, a stretch
of EEG, and the decoder weights.

    python tools/example_reconstruction.py --sub sub-002 \\
        --result results/newlistener/decoder_basic_lam11.json \\
        --out results/newlistener/example_sub-002.npz
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eg606.eval.linear import accumulate, lags_for, predict, zscore  # noqa: E402
from eg606.eval.newlistener import fit, load_trials, window_scores  # noqa: E402
from eg606.paths import derived_dir  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--result", required=True, help="decoder result whose penalty to reuse")
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--eeg-seconds", type=float, default=20.0, help="EEG kept per song")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    res = json.loads(Path(args.result).read_text())
    lam = res["lambda"][args.sub]["onset"]
    trials, targets, cols, fs = load_trials(res.get("variant", "basic"), ["onset"])
    lags = lags_for(fs)
    xtx = xty = None
    for t in trials:
        if t["sub"] != args.sub:
            xtx, xty = accumulate(t["eeg"], t["Y"], xtx, xty, lags)
    W = fit(xtx, xty, {"onset": lam}, cols)

    n_common, win = res["n_common"], int(args.window * fs)
    songs = sorted(targets)
    cand = np.stack([zscore(targets[s]["onset"][:, :n_common]).T for s in songs])   # (12, n, 1)
    out = {"fs": fs, "songs": np.array(songs), "lam": lam, "window_s": args.window,
           "cand": cand[..., 0].astype(np.float32), "weights": W[:, 0].astype(np.float32),
           "n_lags": len(lags)}
    keep = int(args.eeg_seconds * fs)
    for t in (t for t in trials if t["sub"] == args.sub):
        s = t["song"]
        P = predict(t["eeg"], W, len(t["Y"]), lags)
        out[f"pred{s:02d}"] = P[:n_common, 0].astype(np.float32)
        out[f"scores{s:02d}"] = np.array([window_scores(P[a:a + win], cand[:, a:a + win])
                                          for a in range(0, n_common - win + 1, win)])
        out[f"eeg{s:02d}"] = t["eeg"][:, :keep]
    suffix = res.get("variant", "basic")
    f = next(p for p in derived_dir("musin_g").glob(f"{args.sub}_{suffix}.npz")
             if re.match(rf"^{args.sub}_{re.escape(suffix)}$", p.stem))
    out["ch_names"] = np.load(f, allow_pickle=True)["ch_names"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    hits = sum(int(np.argmax(out[f"scores{s:02d}"], 1).tolist().count(songs.index(s)))
               for s in songs)
    total = sum(len(out[f"scores{s:02d}"]) for s in songs)
    print(f"{args.sub}: {hits}/{total} 10 s windows correct; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
