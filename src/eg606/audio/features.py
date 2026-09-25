"""Audio features time-aligned to EEG.

The envelope follows the auditory-EEG convention: sub-band magnitudes, power-law compression,
averaged across bands, then resampled to the EEG rate (Biesmans et al.; ICASSP challenge code).
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

POWER = 0.6      # compression exponent
FEAT_SR = 8000   # common internal rate: MUSIN-G audio is 8 kHz, so speech is matched to it
FMIN, FMAX = 50.0, 3800.0
_CHUNK = 20000   # frames per block, keeps memory bounded on 15-minute recordings


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a PCM wav as mono float64 in [-1, 1]."""
    with wave.open(str(path)) as w:
        n, ch, width, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if dtype == np.uint8:
        x = x - 128.0
    x = x.reshape(-1, ch).mean(axis=1)
    return x / (np.abs(x).max() or 1.0), sr


def to_feat_sr(x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Resample to the common feature rate so every dataset gets identical features."""
    if sr == FEAT_SR:
        return x, sr
    g = np.gcd(FEAT_SR, sr)
    return resample_poly(x, FEAT_SR // g, sr // g), FEAT_SR


def _spectrogram(x: np.ndarray, sr: int, hop: int, win: int):
    """Magnitude spectrogram (frames x freqs) computed in blocks, with its frequency axis."""
    n_frames = 1 + max(0, (len(x) - win)) // hop
    window = np.hanning(win)
    out = []
    for start in range(0, n_frames, _CHUNK):
        stop = min(start + _CHUNK, n_frames)
        idx = np.arange(win)[None, :] + hop * np.arange(start, stop)[:, None]
        out.append(np.abs(np.fft.rfft(x[idx] * window[None, :], axis=1)))
    mag = np.concatenate(out) if out else np.zeros((0, win // 2 + 1))
    return mag, np.fft.rfftfreq(win, 1 / sr)


def envelope(x: np.ndarray, sr: int, fs_out: float, n_bands: int = 28,
             fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    """Compressed broadband envelope at fs_out Hz."""
    x, sr = to_feat_sr(x, sr)
    win = int(round(sr * 0.025)) or 2          # 25 ms window
    hop = max(int(round(sr / (fs_out * 4))), 1)  # 4x oversampled, decimated below
    mag, freqs = _spectrogram(x, sr, hop, win)
    edges = np.linspace(_hz2erb(fmin), _hz2erb(fmax), n_bands + 1)
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (freqs >= _erb2hz(lo)) & (freqs < _erb2hz(hi))
        if sel.any():
            bands.append(mag[:, sel].mean(axis=1))
    env = np.power(np.stack(bands, axis=1), POWER).mean(axis=1)
    return _to_rate(env, sr / hop, fs_out)


def onset_envelope(x: np.ndarray, sr: int, fs_out: float) -> np.ndarray:
    """Half-wave-rectified spectral flux (note onsets) at fs_out Hz."""
    x, sr = to_feat_sr(x, sr)
    win = int(round(sr * 0.025)) or 2
    hop = max(int(round(sr / (fs_out * 4))), 1)
    mag, _ = _spectrogram(x, sr, hop, win)
    flux = np.diff(np.log1p(mag), axis=0, prepend=np.log1p(mag[:1]))
    return _to_rate(np.maximum(flux, 0).mean(axis=1), sr / hop, fs_out)


def _to_rate(sig: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    up, down = int(round(fs_out * 1000)), int(round(fs_in * 1000))
    g = np.gcd(up, down)
    return resample_poly(sig, up // g, down // g)


def _hz2erb(f):  # Glasberg & Moore
    return 21.4 * np.log10(1 + 0.00437 * f)


def _erb2hz(e):
    return (10 ** (e / 21.4) - 1) / 0.00437
