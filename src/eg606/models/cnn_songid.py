"""A GuessTheMusic-style spectrogram CNN, for auditing the protocol rather than beating it.

The leakage audit so far used a regularised linear classifier on band power. The obvious objection
is that a stronger model would find real music structure where a linear one cannot, so the gap
between the within-recording protocol and a held-out listener would close. This is the published
family of model -- short windows, per-channel spectrograms, a small convolutional stack -- run
under exactly the same five splits. If it also identifies songs from pre-song silence, the protocol
is what inflates, not the classifier.

Kept deliberately close to the published recipe: 1 s windows, log-magnitude STFT per channel,
convolution over (frequency, time) with channels as feature maps.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

FS = 64
WIN_S = 1.0
NFFT = 64
HOP = 8


def spectrograms(eeg: np.ndarray, win: int = int(WIN_S * FS)) -> np.ndarray:
    """(channels, time) -> (windows, channels, freq, frames) log-magnitude STFT."""
    c, t = eeg.shape
    n = t // win
    if n == 0:
        return np.zeros((0, c, NFFT // 2 + 1, 1), dtype=np.float32)
    x = torch.from_numpy(np.ascontiguousarray(eeg[:, : n * win], dtype=np.float32))
    x = x.reshape(c, n, win).permute(1, 0, 2).reshape(n * c, win)
    s = torch.stft(x, n_fft=NFFT, hop_length=HOP, win_length=NFFT,
                   window=torch.hann_window(NFFT), return_complex=True, center=True)
    s = torch.log1p(s.abs())
    return s.reshape(n, c, s.shape[-2], s.shape[-1]).numpy()


class SongCNN(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(width * 2, n_classes))

    def forward(self, x):
        return self.net(x)
