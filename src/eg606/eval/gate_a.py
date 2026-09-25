"""Gate A: is there a stimulus-locked music signal that survives a held-out listener?

A linear backward model (the standard mTRF decoder) reconstructs the audio envelope from lagged
EEG. It is trained on all listeners but one and tested on the held-out listener, so nothing about
the test person is in the training set.

The decision task is match-mismatch: given a window of EEG, the reconstructed envelope is compared
with the true envelope and with an imposter taken from the SAME song one second later. Both
candidates share the block, so slow drift cannot separate them; only real time-locking can.
Chance is 50%.

    python -m eg606.eval.gate_a --windows 5,10
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from eg606.paths import derived_dir

log = logging.getLogger(__name__)

FS = 64.0
LAGS = np.arange(0, int(round(0.5 * FS)) + 1)  # 0-500 ms: EEG follows the audio
LAMBDAS = np.logspace(2, 11, 19)
GUARD = 1.0  # s between the matched window and the imposter


def lagged(eeg: np.ndarray, n: int) -> np.ndarray:
    """(channels, time) -> (time, channels*lags), zero-padded at the end."""
    c, t = eeg.shape
    out = np.zeros((n, c * len(LAGS)), dtype=np.float64)
    for i, lag in enumerate(LAGS):
        end = min(n, t - lag)
        if end > 0:
            out[:end, i * c:(i + 1) * c] = eeg[:, lag:lag + end].T
    return out


def predict(eeg: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    """Apply a backward model without materialising the lagged design matrix."""
    c = eeg.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i, lag in enumerate(LAGS):
        end = min(n, eeg.shape[1] - lag)
        if end > 0:
            out[:end] += eeg[:, lag:lag + end].T @ w[i * c:(i + 1) * c]
    return out


def _z(x: np.ndarray, axis=-1) -> np.ndarray:
    m, s = x.mean(axis=axis, keepdims=True), x.std(axis=axis, keepdims=True)
    return (x - m) / np.where(s > 0, s, 1.0)


def subject_stats(path: Path, feats: dict, cache: Path | None = None) -> dict:
    """Accumulate X'X and X'y for one subject, and keep its trials for testing."""
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    xtx = xty = None
    trials = []
    for t in meta:
        song = t["song"]
        eeg = _z(d[f"song{song:02d}"].astype(np.float64))
        env = feats[f"env{song:02d}"].astype(np.float64)
        n = min(eeg.shape[1], len(env))
        X, y = lagged(eeg[:, :n], n), _z(env[:n])
        xtx = X.T @ X if xtx is None else xtx + X.T @ X
        xty = X.T @ y if xty is None else xty + X.T @ y
        trials.append((song, eeg[:, :n], y))
    return {"xtx": xtx, "xty": xty, "trials": trials,
            "silence": {t["song"]: _z(d[f"pre{t['song']:02d}"].astype(np.float64)) for t in meta}}


def target(feats: dict, song: int, kind: str, n: int) -> np.ndarray:
    """The signal the decoder reconstructs: loudness envelope, note onsets, or their sum."""
    env = _z(feats[f"env{song:02d}"].astype(np.float64)[:n])
    if kind == "env":
        return env
    onset = _z(feats[f"onset{song:02d}"].astype(np.float64)[:n])
    return onset if kind == "onset" else _z(env + onset)


def load_or_build(path: Path, feats: dict, kind: str = "env") -> dict:
    """X'X depends only on the EEG, so it is cached; X'y is cheap to recompute per feature."""
    cache = path.parent / "gate_a" / f"{path.stem}_xtx.npz"
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    trials, silence = [], {}
    for t in meta:
        song = t["song"]
        eeg = _z(d[f"song{song:02d}"].astype(np.float64))
        n = min(eeg.shape[1], len(feats[f"env{song:02d}"]))
        trials.append((song, eeg[:, :n], target(feats, song, kind, n)))
        silence[song] = _z(d[f"pre{song:02d}"].astype(np.float64))

    xtx = np.load(cache)["xtx"] if cache.exists() else None
    xty = None
    for _, eeg, y in trials:
        X = lagged(eeg, len(y))
        if xtx is None or not cache.exists():
            xtx = X.T @ X if xtx is None else xtx + X.T @ X
        xty = X.T @ y if xty is None else xty + X.T @ y
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, xtx=xtx)
    return {"xtx": xtx, "xty": xty, "trials": trials, "silence": silence}


def match_mismatch(pred: np.ndarray, env: np.ndarray, win: int, guard: int) -> tuple[int, int]:
    """Compare each window's reconstruction with the true envelope and a same-song imposter.

    Windows are bounded by the reconstruction (which may be a short silence epoch); the imposter
    is drawn from the same song one guard-period later, wrapping to earlier audio if the song ends.
    """
    hits = total = 0
    for start in range(0, len(pred) - win + 1, win):
        p = pred[start:start + win]
        true = env[start:start + win]
        if len(true) < win or p.std() == 0 or true.std() == 0:
            continue
        imp_start = start + win + guard
        if imp_start + win > len(env):           # near the end: take the imposter before the window
            imp_start = start - win - guard
            if imp_start < 0:
                continue
        imp = env[imp_start:imp_start + win]
        if len(imp) < win or imp.std() == 0:
            continue
        hits += int(np.corrcoef(p, true)[0, 1] > np.corrcoef(p, imp)[0, 1])
        total += 1
    return hits, total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.eval.gate_a")
    ap.add_argument("--windows", default="5,10", help="match-mismatch window lengths in seconds")
    ap.add_argument("--subjects", default="all")
    ap.add_argument("--silence-gap", type=float, default=0.0,
                    help="drop the last S seconds of the pre-song silence (zero-phase filters smear "
                         "the song onset backward by up to half their length, ~3.3 s at 0.5 Hz)")
    ap.add_argument("--variant", default="none",
                    help="which preprocessed set: none | basic | ica | bs64 | basic_bs64 | ica_bs64")
    ap.add_argument("--feature", default="env", choices=["env", "onset", "both"],
                    help="what the decoder reconstructs")
    ap.add_argument("--out", default=None, help="write per-subject results as JSON")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    windows = [float(w) for w in args.windows.split(",")]

    root = derived_dir("musin_g")
    feats = dict(np.load(root / "audio_features.npz"))
    # Variants live side by side (sub-001.npz, sub-001_basic.npz, sub-001_ica_bs64.npz ...) and have
    # different channel counts, so select exactly one set rather than globbing loosely.
    suffix = "" if args.variant == "none" else f"_{args.variant}"
    pattern = re.compile(rf"^sub-\d+{re.escape(suffix)}$")
    paths = sorted(p for p in root.glob("sub-*.npz") if pattern.match(p.stem))
    if not paths:
        log.error("no files for variant %r in %s", args.variant, root)
        return 1
    if args.subjects != "all":
        keep = set(args.subjects.split(","))
        paths = [p for p in paths if p.stem.split("_")[0] in keep]
    log.info("%d subjects, %d lags, feature=%s, variant=%s", len(paths), len(LAGS),
             args.feature, args.variant)

    stats = {}
    for p in paths:
        stats[p.stem] = load_or_build(p, feats, args.feature)
        log.info("accumulated %s", p.stem)

    results = {}
    for held in stats:
        others = [s for s in stats if s != held]
        # inner selection: 4 training listeners act as validation
        val = others[:4]
        inner_tr = [s for s in others if s not in val]
        xtx = sum(stats[s]["xtx"] for s in inner_tr)
        xty = sum(stats[s]["xty"] for s in inner_tr)
        best_lam, best_r = None, -np.inf
        for lam in LAMBDAS:
            w = cho_solve(cho_factor(xtx + lam * np.eye(len(xtx))), xty)
            rs = [np.corrcoef(predict(e, w, len(y)), y)[0, 1]
                  for s in val for _, e, y in stats[s]["trials"]]
            if np.mean(rs) > best_r:
                best_lam, best_r = lam, np.mean(rs)

        xtx = sum(stats[s]["xtx"] for s in others)
        xty = sum(stats[s]["xty"] for s in others)
        w = cho_solve(cho_factor(xtx + best_lam * np.eye(len(xtx))), xty)

        rs, mm = [], {f"{win:g}s": [0, 0] for win in windows}
        sil = {f"{win:g}s": [0, 0] for win in windows}
        for song, eeg, y in stats[held]["trials"]:
            pred = predict(eeg, w, len(y))
            rs.append(np.corrcoef(pred, y)[0, 1])
            s_eeg = stats[held]["silence"][song]
            if args.silence_gap > 0:
                s_eeg = s_eeg[:, : max(0, s_eeg.shape[1] - int(args.silence_gap * FS))]
            s_pred = predict(s_eeg, w, s_eeg.shape[1])
            for win in windows:
                n, g = int(win * FS), int(GUARD * FS)
                h, t = match_mismatch(pred, y, n, g)
                mm[f"{win:g}s"][0] += h; mm[f"{win:g}s"][1] += t
                # control N1: silence before the song, scored against the song envelope
                h, t = match_mismatch(s_pred, y, n, g)
                sil[f"{win:g}s"][0] += h; sil[f"{win:g}s"][1] += t

        results[held] = {
            "r_mean": float(np.mean(rs)),
            "mm": {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in mm.items()},
            "silence_mm": {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in sil.items()},
            "lambda": float(best_lam),
        }
        log.info("%s  r=%.4f  %s  silence=%s  lam=%.0e", held, results[held]["r_mean"],
                 {k: round(v, 3) for k, v in results[held]["mm"].items()},
                 {k: round(v, 3) for k, v in results[held]["silence_mm"].items()}, best_lam)

    print(f"\n=== leave-one-subject-out summary, feature={args.feature}, "
          f"variant={args.variant} (chance = 0.5)")
    print(f"envelope r: mean {np.mean([r['r_mean'] for r in results.values()]):.4f}")
    for win in windows:
        k = f"{win:g}s"
        acc = np.array([r["mm"][k] for r in results.values()])
        sil = np.array([r["silence_mm"][k] for r in results.values()])
        above = (acc > 0.5).sum()
        print(f"match-mismatch {k}: mean {acc.mean():.3f} +/- {acc.std():.3f}  "
              f"({above}/{len(acc)} listeners above chance) | silence control {sil.mean():.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
