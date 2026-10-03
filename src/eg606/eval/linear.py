"""Shared pieces of the linear (backward-model) evaluation.

A backward model reconstructs an audio feature from lagged EEG. Fitting is closed-form ridge
regression accumulated as X'X and X'y, so many listeners can be pooled without ever holding the
full design matrix in memory.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

FS = 64.0
LAGS = np.arange(0, int(round(0.5 * FS)) + 1)  # 0-500 ms: EEG follows the audio
GUARD = 1.0                                     # s between a window and its imposter


def zscore(x: np.ndarray, axis=-1) -> np.ndarray:
    m, s = x.mean(axis=axis, keepdims=True), x.std(axis=axis, keepdims=True)
    return (x - m) / np.where(s > 0, s, 1.0)


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


def accumulate(eeg: np.ndarray, y: np.ndarray, xtx=None, xty=None):
    """Add one trial to the normal equations."""
    X = lagged(eeg, len(y))
    return (X.T @ X if xtx is None else xtx + X.T @ X,
            X.T @ y if xty is None else xty + X.T @ y)


def solve(xtx: np.ndarray, xty: np.ndarray, lam: float) -> np.ndarray:
    return cho_solve(cho_factor(xtx + lam * np.eye(len(xtx))), xty)


def deranged(segments, rng, alt_targets=None):
    """Break the correspondence between each EEG segment and its audio, changing nothing else.

    Match-mismatch can be solved without reading the EEG at all if the scoring prefers one
    candidate systematically, and the imposter is always drawn later in the same recording, so the
    two candidates are not interchangeable. Anything above chance under this control is the task
    leaking rather than the brain.

    Which swap is valid depends on what was held out. A held-out listener leaves the test set
    spanning many songs, so permuting EEG among the segments suffices. A held-out *song* leaves
    every segment sharing one stimulus, where permuting changes nothing and the control silently
    reproduces the real result; there the audio must come from elsewhere, via `alt_targets`.
    """
    if len(segments) < 2:
        return []
    out = []
    if alt_targets:
        for eeg, _ in segments:
            y = alt_targets[rng.integers(len(alt_targets))]
            n = min(eeg.shape[1], len(y))
            out.append((eeg[:, :n], y[:n]))
        return out
    idx = rng.permutation(len(segments))
    for i in range(len(idx)):                 # a derangement: no target keeps its own EEG
        if idx[i] == i:
            j = (i + 1) % len(idx)
            idx[i], idx[j] = idx[j], idx[i]
    for i, (_, y) in enumerate(segments):
        eeg = segments[idx[i]][0]
        n = min(eeg.shape[1], len(y))
        out.append((eeg[:, :n], y[:n]))
    return out


def match_mismatch(pred: np.ndarray, env: np.ndarray, win: int,
                   guard: int = int(GUARD * FS), rng=None) -> tuple[int, int]:
    """Same-song imposter one guard-period later; falls back to earlier audio at a song's end."""
    hits = total = 0
    for start in range(0, len(pred) - win + 1, win):
        p, true = pred[start:start + win], env[start:start + win]
        if len(true) < win or p.std() == 0 or true.std() == 0:
            continue
        # The imposter is normally taken after the window. That makes the two candidates
        # systematically different in position, which a scorer with any positional bias can
        # exploit; passing an rng draws the side at random instead.
        later = True if rng is None else bool(rng.integers(2))
        imp_start = start + win + guard if later else start - win - guard
        if imp_start < 0 or imp_start + win > len(env):
            imp_start = start - win - guard if later else start + win + guard
            if imp_start < 0 or imp_start + win > len(env):
                continue
        imp = env[imp_start:imp_start + win]
        if len(imp) < win or imp.std() == 0:
            continue
        hits += int(np.corrcoef(p, true)[0, 1] > np.corrcoef(p, imp)[0, 1])
        total += 1
    return hits, total
