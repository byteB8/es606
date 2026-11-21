"""Which song is a new listener hearing? Song identification with a time-locked decoder.

The project's single objective: name the song (12-way, MUSIN-G) from the EEG of a listener whose
data the model never saw in training. The decoder is the linear backward model of eval/linear.py.
It reconstructs the music's note-onset curve from 0-500 ms of EEG, is fitted on the other
listeners, and names the song whose onset curve correlates best with the reconstruction over the
same stretch of time. Classifier baselines given exactly the same test decisions are in
eval/newlistener_baselines.py.

Splits
  loso       new listener: fit on 19 listeners, test on the 20th, every listener in turn
  loso_song  new listener AND new song: each test trial is decoded by a model that never saw that
             listener or that song (one fit per trial)

Every decision uses the first `n_common` samples of a trial (the shortest song, about 100 s), so
all 12 candidates exist at every moment and chance is exactly 1/12. A decision window of W
seconds steps over that stretch without overlap; "all" uses the whole stretch once.

Control: is the EEG doing the work, in time with the music?
  shift  the test EEG rolled in time by at least 10 s: same listener, recording and song, only the
         timing destroyed. Chance 1/12.
  A "wrong song" control (score song A with the EEG of song B) is deliberately absent: with every
  song a candidate and every song a label, the guess made from B's EEG is right for exactly one of
  the eleven labels it is scored against, so that control equals 1/11 by construction and cannot
  detect anything. Balanced candidates at the same moment of every song also rule out the
  match-mismatch leaks (an imposter from later in the song, short trials) by design.

Music targets: how much of the music's frequency content the decoder is asked to follow
  onset         spectral flux averaged over every audio frequency (0-4 kHz; the files are 8 kHz)
  onset_lt500, onset_lt1k, onset_lt2k
                the same, from the audio below 0.5, 1 or 2 kHz only
  onset_8band   one onset curve per ERB band (8 bands, 50 Hz to 4 kHz), matched band by band
  env           the loudness envelope, for reference

    python -m eg606.eval.newlistener --variant basic --splits loso,loso_song \\
        --out results/newlistener/decoder_basic.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.signal import butter, filtfilt, hilbert, resample_poly

from eg606.audio.features import envelope, erb_bands, onset_bands, read_wav
from eg606.eval.linear import accumulate, lags_for, predict, zscore
from eg606.eval.stats import boot_ci, sign_flip_p
from eg606.paths import derived_dir, raw_dir

log = logging.getLogger(__name__)

LAMBDAS = np.logspace(3, 9, 13)
N_VAL = 3                     # validation listeners for choosing the ridge penalty
SHIFT_MIN_S = 10.0            # the time-shift control moves the EEG by at least this much
GAMMA = (32.0, 45.0)          # Hz, for the gamma-envelope EEG
TARGETS = {
    "onset": [(0.0, np.inf)],
    "onset_lt500": [(0.0, 500.0)],
    "onset_lt1k": [(0.0, 1000.0)],
    "onset_lt2k": [(0.0, 2000.0)],
    "onset_8band": erb_bands(8),
    "env": None,
}
CONDITIONS = ("real", "shift")


# ----------------------------------------------------------------------------- data

def song_targets(kinds: list[str], fs: float) -> dict[int, dict[str, np.ndarray]]:
    """Each song's targets at fs Hz, as (bands, time) arrays, computed from the stimulus files."""
    root = raw_dir("musin_g") / "Code" / "ESongs"
    out = {}
    for song in range(1, 13):
        x, sr = read_wav(root / f"{song}.esh.wav")
        out[song] = {k: (envelope(x, sr, fs)[None] if k == "env"
                         else onset_bands(x, sr, fs, TARGETS[k])) for k in kinds}
    return out


def gamma_envelope(eeg: np.ndarray, fs: float) -> tuple[np.ndarray, float]:
    """Amplitude envelope of the 32-45 Hz EEG, brought to 64 Hz like every other input."""
    if fs < 2 * GAMMA[1] + 10:
        raise ValueError(f"gamma envelope needs EEG above {GAMMA[1]} Hz; this file is at {fs} Hz")
    b, a = butter(4, GAMMA, btype="band", fs=fs)
    env = np.abs(hilbert(filtfilt(b, a, eeg, axis=-1), axis=-1))
    step = int(round(fs / 64.0))
    return resample_poly(env, 1, step, axis=-1), fs / step


def load_trials(variant: str, kinds: list[str], eeg_mode: str = "raw", max_subjects=None):
    """One dict per (listener, song): z-scored EEG and the stacked, z-scored targets."""
    root = derived_dir("musin_g")
    suffix = "" if variant == "none" else f"_{variant}"
    pattern = re.compile(rf"^sub-\d+{re.escape(suffix)}$")
    paths = sorted(p for p in root.glob("sub-*.npz") if pattern.match(p.stem))
    if not paths:
        raise FileNotFoundError(f"no sub-*{suffix}.npz in {root}")
    paths = paths[:max_subjects] if max_subjects else paths

    trials, targets, fs = [], None, None
    for p in paths:
        d = np.load(p, allow_pickle=True)
        sub = p.stem.split("_")[0]
        for m in json.loads(str(d["meta"])):
            song = m["song"]
            eeg, f = d[f"song{song:02d}"].astype(np.float64), float(d["fs"])
            if eeg_mode == "gamma_env":
                eeg, f = gamma_envelope(eeg, f)
            if targets is None:
                fs, targets = f, song_targets(kinds, f)
            n = min([eeg.shape[1]] + [targets[song][k].shape[1] for k in kinds])
            Y = np.concatenate([zscore(targets[song][k][:, :n]) for k in kinds]).T
            trials.append({"sub": sub, "song": song,
                           "eeg": zscore(eeg[:, :n]).astype(np.float32), "Y": Y})
        log.info("loaded %s (%d trials so far)", p.name, len(trials))

    sizes = [targets[1][k].shape[0] for k in kinds]
    stops = np.cumsum(sizes)
    cols = {k: slice(int(stop - size), int(stop)) for k, size, stop in zip(kinds, sizes, stops)}
    return trials, targets, cols, fs


# ----------------------------------------------------------------------------- fitting

def _add(acc: dict, key, xtx, xty):
    if key in acc:
        acc[key][0] += xtx
        acc[key][1] += xty
    else:
        acc[key] = [xtx.copy(), xty.copy()]


def _ridge(xtx: np.ndarray, lam: float):
    a = xtx.copy()
    a[np.diag_indices_from(a)] += lam
    return cho_factor(a, check_finite=False)


def fit(xtx, xty, lam_by_kind: dict, cols: dict) -> np.ndarray:
    """Ridge solution for every target column, one Cholesky per distinct penalty."""
    W = np.zeros_like(xty)
    for lam in sorted(set(lam_by_kind.values())):
        idx = np.concatenate([np.arange(cols[k].start, cols[k].stop)
                              for k in cols if lam_by_kind[k] == lam])
        W[:, idx] = cho_solve(_ridge(xtx, lam), xty[:, idx], check_finite=False)
    return W


def corr_bands(p: np.ndarray, y: np.ndarray) -> float:
    """Pearson r between matching columns of two (time, bands) arrays, averaged over bands."""
    pc, yc = p - p.mean(0), y - y.mean(0)
    den = np.sqrt((pc ** 2).sum(0) * (yc ** 2).sum(0))
    r = np.where(den > 0, (pc * yc).sum(0) / np.where(den > 0, den, 1.0), 0.0)
    return float(r.mean())


def select_lambdas(xtx, xty, val_trials, cols, lags, lambdas=LAMBDAS) -> dict:
    """Per target, the penalty that best reconstructs the validation listeners' targets."""
    best = {k: (lambdas[len(lambdas) // 2], -np.inf) for k in cols}
    for lam in lambdas:
        W = cho_solve(_ridge(xtx, lam), xty, check_finite=False)
        rs = {k: [] for k in cols}
        for t in val_trials:
            P = predict(t["eeg"], W, len(t["Y"]), lags)
            for k, sl in cols.items():
                rs[k].append(corr_bands(P[:, sl], t["Y"][:, sl]))
        for k in cols:
            r = float(np.mean(rs[k]))
            if r > best[k][1]:
                best[k] = (float(lam), r)
    return {k: v[0] for k, v in best.items()}


# ----------------------------------------------------------------------------- scoring

def window_scores(p: np.ndarray, C: np.ndarray) -> np.ndarray:
    """p (win, bands) against candidates C (songs, win, bands): mean Fisher-z r per song."""
    pc = p - p.mean(0)
    Cc = C - C.mean(1, keepdims=True)
    num = np.einsum("wb,swb->sb", pc, Cc)
    den = np.sqrt((pc ** 2).sum(0))[None] * np.sqrt((Cc ** 2).sum(1))
    r = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return np.arctanh(np.clip(r, -0.999999, 0.999999)).mean(1)


def identify(P, cand, songs, label, win, n_common):
    """Non-overlapping decisions over the first n_common samples. Returns (hits, decisions, guesses)."""
    hits, guesses = 0, []
    for a in range(0, n_common - win + 1, win):
        g = songs[int(np.argmax(window_scores(P[a:a + win], cand[:, a:a + win])))]
        hits += int(g == label)
        guesses.append(int(g))
    return hits, len(guesses), guesses


class Scorer:
    """Accumulates real and control decisions per listener, target and window."""

    def __init__(self, cols, cand, songs, wins, n_common, lags, fs, n_shifts, seed):
        self.cols, self.cand, self.songs, self.wins = cols, cand, songs, wins
        self.n_common, self.lags, self.fs, self.n_shifts = n_common, lags, fs, n_shifts
        self.seed = seed
        self.rec: dict = {}

    def _slot(self, sub, kind):
        return self.rec.setdefault(kind, {}).setdefault(sub, {
            "r": [], "r_shift": [], "guess_all": {},
            "win": {w: {c: [0, 0] for c in CONDITIONS} for w in self.wins}})

    def _tally(self, sub, kind, cond, P, label):
        slot = self._slot(sub, kind)
        for w, n in self.wins.items():
            h, t, g = identify(P, self.cand[kind], self.songs, label, n, self.n_common)
            slot["win"][w][cond][0] += h
            slot["win"][w][cond][1] += t
            if cond == "real" and w == "all":
                slot["guess_all"][str(label)] = g[0]

    def trial(self, t, W):
        """Score one test trial and its time-shift control; returns the reconstruction."""
        n = len(t["Y"])
        P = predict(t["eeg"], W, n, self.lags)
        rng = np.random.default_rng([self.seed, int(t["sub"][4:]), t["song"]])
        lo = int(SHIFT_MIN_S * self.fs)
        shifts = rng.integers(lo, n - lo, self.n_shifts) if n > 2 * lo else []
        Ps = [predict(np.roll(t["eeg"], int(s), axis=1), W, n, self.lags) for s in shifts]
        for k, sl in self.cols.items():
            slot = self._slot(t["sub"], k)
            slot["r"].append(corr_bands(P[:, sl], t["Y"][:, sl]))
            self._tally(t["sub"], k, "real", P[:, sl], t["song"])
            for p in Ps:
                slot["r_shift"].append(corr_bands(p[:, sl], t["Y"][:, sl]))
                self._tally(t["sub"], k, "shift", p[:, sl], t["song"])
        return P


# ----------------------------------------------------------------------------- statistics

def summarize(per_listener: dict, n_songs: int, conds=CONDITIONS) -> dict:
    """Per window: listener-level accuracy, bootstrap CI, decisions and sign-flip tests."""
    chance = {"real": 1 / n_songs, "shift": 1 / n_songs}
    first = next(iter(per_listener.values()))
    out = {}
    for w in first["win"]:
        row, acc = {}, {}
        for c in conds:
            pairs = [v["win"][w][c] for v in per_listener.values()]
            if not any(p[1] for p in pairs):
                continue
            acc[c] = np.array([h / n if n else np.nan for h, n in pairs])
            a = acc[c][~np.isnan(acc[c])]
            row[c] = {"mean": float(a.mean()), "ci": boot_ci(a), "chance": chance[c],
                      "decisions": int(sum(p[1] for p in pairs)),
                      "listeners_above_chance": int((a > chance[c]).sum()), "listeners": len(a)}
        if "real" in acc:
            row["p_vs_chance"] = sign_flip_p(acc["real"] - chance["real"])[0]
            if "shift" in acc:
                row["p_vs_shift"] = sign_flip_p(acc["real"] - acc["shift"])[0]
        out[w] = row
    return out


def print_table(title: str, summary: dict, conds=CONDITIONS):
    wins = list(next(iter(summary.values())).keys())
    print(f"\n=== {title}")
    print(f"{'target':<14}{'':<7}" + "".join(f"{w + ('s' if w != 'all' else ''):>9}" for w in wins))
    for kind, s in summary.items():
        for c in conds:
            if c not in s[wins[0]]:
                continue
            label = kind if c == "real" else ""
            print(f"{label:<14}{c:<7}" + "".join(f"{s[w][c]['mean']:9.3f}" for w in wins))
        print(f"{'':<14}{'p':<7}" + "".join(f"{s[w].get('p_vs_chance', np.nan):9.1g}" for w in wins))
    first = next(iter(summary.values()))
    print(f"{'decisions':<21}" + "".join(f"{first[w]['real']['decisions']:9d}" for w in wins))


# ----------------------------------------------------------------------------- main

def run(trials, targets, cols, fs, splits, windows, n_shifts, seed, lambdas=LAMBDAS) -> dict:
    lags = lags_for(fs)
    subs = sorted({t["sub"] for t in trials})
    songs = sorted({t["song"] for t in trials})
    n_common = min(len(t["Y"]) for t in trials)
    wins = {}
    for w in windows:
        n = n_common if w == "all" else int(round(float(w) * fs))
        if n <= n_common:
            wins[w] = n
    cand = {k: np.stack([zscore(targets[s][k][:, :n_common]).T for s in songs]) for k in cols}
    log.info("%d listeners, %d songs, %d channels x %d lags, fs %g Hz, n_common %d (%.1f s)",
             len(subs), len(songs), trials[0]["eeg"].shape[0], len(lags), fs, n_common,
             n_common / fs)

    t0 = time.time()
    by_sub, by_song = {}, {}
    for t in trials:
        xtx, xty = accumulate(t["eeg"], t["Y"], lags=lags)
        _add(by_sub, t["sub"], xtx, xty)
        if "loso_song" in splits:
            _add(by_song, t["song"], xtx, xty)
    total = [sum(v[0] for v in by_sub.values()), sum(v[1] for v in by_sub.values())]
    log.info("normal equations ready (%.0f s)", time.time() - t0)

    lam = {}
    for L in subs:
        others = [s for s in subs if s != L]
        n_val = max(1, min(N_VAL, len(others) // 3))
        val = others[:n_val]
        xtx = total[0] - by_sub[L][0] - sum(by_sub[g][0] for g in val)
        xty = total[1] - by_sub[L][1] - sum(by_sub[g][1] for g in val)
        lam[L] = select_lambdas(xtx, xty, [t for t in trials if t["sub"] in val], cols, lags,
                                lambdas)
        log.info("penalty %s %s", L, {k: f"{v:.0e}" for k, v in lam[L].items()})
    edge = {k: sum(lam[L][k] in (lambdas[0], lambdas[-1]) for L in subs) for k in cols}
    if any(edge.values()):
        log.warning("penalty at the edge of the grid (folds per target): %s", edge)

    out = {"fs": fs, "n_lags": len(lags), "n_common": n_common, "n_common_s": n_common / fs,
           "songs": songs, "windows": wins, "lambda": lam,
           "lambda_grid": [float(lambdas[0]), float(lambdas[-1]), len(lambdas)],
           "lambda_at_edge": edge, "splits": {}}
    for split in splits:
        sc = Scorer(cols, cand, songs, wins, n_common, lags, fs, n_shifts, seed)
        for L in subs:
            test = [t for t in trials if t["sub"] == L]
            if split == "loso":
                W = fit(total[0] - by_sub[L][0], total[1] - by_sub[L][1], lam[L], cols)
                for t in test:
                    sc.trial(t, W)
            elif split == "loso_song":
                for t in test:
                    own = accumulate(t["eeg"], t["Y"], lags=lags)
                    s = t["song"]
                    xtx = total[0] - by_sub[L][0] - by_song[s][0] + own[0]
                    xty = total[1] - by_sub[L][1] - by_song[s][1] + own[1]
                    sc.trial(t, fit(xtx, xty, lam[L], cols))
            else:
                raise ValueError(f"unknown split {split}")
            r = {k: np.mean(sc.rec[k][L]["r"]) for k in cols}
            a = {k: sc.rec[k][L]["win"]["all"]["real"] for k in cols} if "all" in wins else {}
            log.info("%s %s r=%s whole-song hits=%s (%.0f s)", split, L,
                     {k: round(v, 3) for k, v in r.items()}, {k: v[0] for k, v in a.items()},
                     time.time() - t0)
        confusion = {}
        if "all" in wins:
            for k in cols:
                m = np.zeros((len(songs), len(songs)), dtype=int)
                for v in sc.rec[k].values():
                    for true, guess in v["guess_all"].items():
                        m[songs.index(int(true)), songs.index(guess)] += 1
                confusion[k] = m.tolist()
        out["splits"][split] = {
            "per_listener": sc.rec, "confusion_all": confusion,
            "summary": {k: summarize(sc.rec[k], len(songs)) for k in cols}}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eg606.eval.newlistener")
    ap.add_argument("--variant", default="basic", help="cached EEG, e.g. basic or basic_fs128lp45")
    ap.add_argument("--eeg", default="raw", choices=["raw", "gamma_env"])
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--splits", default="loso,loso_song")
    ap.add_argument("--windows", default="2,4,10,20,30,50,all")
    ap.add_argument("--shifts", type=int, default=5, help="time-shift controls per trial")
    ap.add_argument("--max-subjects", type=int, default=None, help="smoke tests only")
    ap.add_argument("--lambdas", default="3,9,13",
                    help="ridge grid as log10(min),log10(max),count (default 1e3..1e9, 13 values)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    lo, hi, n = args.lambdas.split(",")
    lambdas = np.logspace(float(lo), float(hi), int(n))

    kinds = args.targets.split(",")
    unknown = set(kinds) - set(TARGETS)
    if unknown:
        ap.error(f"unknown targets {sorted(unknown)}")
    trials, targets, cols, fs = load_trials(args.variant, kinds, args.eeg, args.max_subjects)
    res = run(trials, targets, cols, fs, args.splits.split(","), args.windows.split(","),
              args.shifts, args.seed, lambdas)
    res.update({"variant": args.variant, "eeg": args.eeg, "targets": kinds, "seed": args.seed})

    for split, r in res["splits"].items():
        print_table(f"new listener song ID, split={split}, variant={args.variant}, "
                    f"eeg={args.eeg}, chance 1/12 = 0.083", r["summary"])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1, default=float))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
