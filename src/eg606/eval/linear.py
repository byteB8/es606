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


def match_mismatch(pred: np.ndarray, env: np.ndarray, win: int,
                   guard: int = int(GUARD * FS)) -> tuple[int, int]:
    """Same-song imposter one guard-period later; falls back to earlier audio at a song's end."""
    hits = total = 0
    for start in range(0, len(pred) - win + 1, win):
        p, true = pred[start:start + win], env[start:start + win]
        if len(true) < win or p.std() == 0 or true.std() == 0:
            continue
        imp_start = start + win + guard
        if imp_start + win > len(env):
            imp_start = start - win - guard
            if imp_start < 0:
                continue
        imp = env[imp_start:imp_start + win]
        if len(imp) < win or imp.std() == 0:
            continue
        hits += int(np.corrcoef(p, true)[0, 1] > np.corrcoef(p, imp)[0, 1])
        total += 1
    return hits, total
