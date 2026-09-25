"""Contrastive EEG-audio model: does this EEG window match this audio window?

Design follows what the linear experiments showed:
  * the target is note onsets (spectral flux), which beat the loudness envelope by 4-12 points;
  * training and scoring use the match-mismatch task, so numbers compare directly with the
    linear baseline (0.602 at 10 s, leave-one-listener-out);
  * both towers output one embedding per short window, and longer windows are scored by averaging
    sub-window similarities - evidence accumulation rather than a longer receptive field.

The EEG tower is deliberately small (a spatial mixer then dilated temporal convolutions): with
8 hours of music EEG, capacity is not the binding constraint.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGEncoder(nn.Module):
    def __init__(self, n_channels: int = 64, width: int = 128, dim: int = 128, dropout: float = 0.1):
        super().__init__()
        # 1. mix channels (a learned spatial filter, like the first stage of EEGNet/VLAAI)
        self.spatial = nn.Conv1d(n_channels, width, kernel_size=1)
        self.norm0 = nn.BatchNorm1d(width)
        # 2. temporal context at several scales; 0-500 ms is where auditory tracking lives
        self.blocks = nn.ModuleList()
        for dilation in (1, 2, 4, 8):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(width, width, kernel_size=5, padding=2 * dilation, dilation=dilation),
                nn.BatchNorm1d(width),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
        self.head = nn.Conv1d(width, dim, kernel_size=1)

    def forward(self, x):                      # (B, C, T) -> (B, dim)
        h = F.gelu(self.norm0(self.spatial(x)))
        for block in self.blocks:
            h = h + block(h)
        h = self.head(h)
        return F.normalize(h.mean(dim=-1), dim=-1)


class AudioEncoder(nn.Module):
    def __init__(self, n_features: int = 1, width: int = 64, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, width, kernel_size=5, padding=2), nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width, kernel_size=5, padding=4, dilation=2), nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, dim, kernel_size=1),
        )

    def forward(self, x):                      # (B, F, T) -> (B, dim)
        return F.normalize(self.net(x).mean(dim=-1), dim=-1)


class Matcher(nn.Module):
    """Two towers and a temperature; similarity is cosine."""

    def __init__(self, n_channels: int = 64, dim: int = 128, **kw):
        super().__init__()
        self.eeg = EEGEncoder(n_channels=n_channels, dim=dim, **kw)
        self.audio = AudioEncoder(dim=dim)
        self.log_temp = nn.Parameter(torch.tensor(2.6))   # ~1/0.07

    def forward(self, eeg, audio):
        return self.eeg(eeg), self.audio(audio)

    def similarity(self, eeg, audio):
        e, a = self(eeg, audio)
        return (e * a).sum(-1)

    def pair_score(self, eeg, audio):
        return self.similarity(eeg, audio)

    def loss(self, eeg, audio):
        """Symmetric InfoNCE over the batch. Negatives are the other windows in the batch, which
        are drawn from the same recordings, so drift cannot separate them."""
        e, a = self(eeg, audio)
        logits = e @ a.T * self.log_temp.exp().clamp(max=100.0)
        target = torch.arange(len(e), device=e.device)
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


@torch.no_grad()
def match_mismatch_accuracy(model, eeg, target, window: int, sub: int, guard: int,
                            device, batch: int = 256) -> tuple[int, int]:
    """Score a whole trial: matched audio vs an imposter from the same trial, `guard` samples later.

    A long window is scored by averaging similarities over `sub`-sample sub-windows, which is how
    the linear baseline accumulates evidence.
    """
    model.eval()
    n = min(eeg.shape[1], target.shape[-1])
    hits = total = 0
    starts = list(range(0, n - window + 1, window))
    for s in starts:
        imp = s + window + guard
        if imp + window > n:
            imp = s - window - guard
            if imp < 0:
                continue
        offs = list(range(0, window - sub + 1, sub))
        if not offs:
            continue
        eeg_b = torch.stack([eeg[:, s + o: s + o + sub] for o in offs]).to(device)
        aud_t = torch.stack([target[..., s + o: s + o + sub] for o in offs]).to(device)
        aud_i = torch.stack([target[..., imp + o: imp + o + sub] for o in offs]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            sim_t = model.pair_score(eeg_b, aud_t).float().mean()
            sim_i = model.pair_score(eeg_b, aud_i).float().mean()
        hits += int(sim_t > sim_i)
        total += 1
    return hits, total


# ----------------------------------------------------------------------------------------------
# v2: time-resolved matching
#
# v1 averaged each tower over the whole window before comparing, which discards the moment-to-
# moment alignment the task depends on (it matched window-level statistics; it overfit music and
# barely learned speech). v2 keeps a short sequence of embeddings (8 Hz) and scores a pair by the
# mean over time of the per-step cosine similarity - the learned analogue of the correlation the
# linear decoder computes.
# ----------------------------------------------------------------------------------------------

class SeqEEGEncoder(nn.Module):
    def __init__(self, n_channels: int = 64, width: int = 128, dim: int = 64,
                 dropout: float = 0.2, pool: int = 8):
        super().__init__()
        self.spatial = nn.Conv1d(n_channels, width, kernel_size=1)
        self.norm0 = nn.BatchNorm1d(width)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(width, width, kernel_size=5, padding=2 * d, dilation=d),
                nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(dropout))
            for d in (1, 2, 4, 8))                      # receptive field ~0.95 s at 64 Hz
        self.head = nn.Conv1d(width, dim, kernel_size=1)
        self.pool = nn.AvgPool1d(pool)

    def forward(self, x):                              # (B, C, T) -> (B, dim, T/pool)
        h = F.gelu(self.norm0(self.spatial(x)))
        for block in self.blocks:
            h = h + block(h)
        return F.normalize(self.pool(self.head(h)), dim=1)


class SeqAudioEncoder(nn.Module):
    def __init__(self, n_features: int = 1, width: int = 64, dim: int = 64, pool: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, width, kernel_size=5, padding=2), nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width, kernel_size=5, padding=4, dilation=2), nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, dim, kernel_size=1))
        self.pool = nn.AvgPool1d(pool)

    def forward(self, x):
        return F.normalize(self.pool(self.net(x)), dim=1)


class TimeMatcher(nn.Module):
    def __init__(self, n_channels: int = 64, dim: int = 64, channel_drop: float = 0.1, **kw):
        super().__init__()
        self.eeg = SeqEEGEncoder(n_channels=n_channels, dim=dim, **kw)
        self.audio = SeqAudioEncoder(dim=dim)
        self.log_temp = nn.Parameter(torch.tensor(2.6))
        self.channel_drop = channel_drop

    def _drop_channels(self, eeg):
        if not self.training or self.channel_drop <= 0:
            return eeg
        keep = (torch.rand(eeg.shape[0], eeg.shape[1], 1, device=eeg.device) > self.channel_drop)
        return eeg * keep / (1 - self.channel_drop)

    def embed(self, eeg, audio):
        e, a = self.eeg(self._drop_channels(eeg)), self.audio(audio)
        n = min(e.shape[-1], a.shape[-1])
        return e[..., :n], a[..., :n]

    def loss(self, eeg, audio):
        e, a = self.embed(eeg, audio)
        logits = torch.einsum("bdt,cdt->bc", e, a) / e.shape[-1] * self.log_temp.exp().clamp(max=100.0)
        target = torch.arange(len(e), device=e.device)
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))

    def pair_score(self, eeg, audio):
        e, a = self.embed(eeg, audio)
        return (e * a).sum(1).mean(-1)
