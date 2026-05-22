"""BMnet — MNASNet-backed body-measurement regressor.

Architecture (paper §3.1): an MNASNet backbone (depth multiplier 1)
consumes a 3-channel input — channel 0 is the front|side silhouette pair
concatenated side by side, channels 1 and 2 are constant-valued planes
carrying the (standardized) height and weight. MNASNet's 1280-wide
feature stack is global-average-pooled and fed to an MLP with one
128-neuron hidden layer and 14 measurement outputs.

The network regresses *standardized* measurements; ``Standardizer`` (held
in the checkpoint, not a Module) maps to and from centimetres so the
training loss and the reported error are both in cm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mnasnet1_0


class BMnet(nn.Module):
    """Front+side silhouette + H/W  →  14 standardized measurements."""

    def __init__(self, n_out: int = 14, hidden: int = 128,
                 dropout: float = 0.2) -> None:
        super().__init__()
        backbone = mnasnet1_0(weights=None)
        # ``layers`` is the full conv stack ending in 1280 channels; the
        # stock classifier (Linear 1280→1000) is dropped.
        self.features = backbone.layers
        self.head = nn.Sequential(
            nn.Linear(1280, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)            # (B, 1280, h', w')
        f = f.mean(dim=[2, 3])          # global average pool
        return self.head(f)


@dataclass
class Standardizer:
    """Per-channel mean/std for measurements and the H/W input pair.

    Stored in the checkpoint so ``predict`` reproduces the exact training
    normalization without the BodyM csvs."""

    meas_mean: np.ndarray   # (14,) cm
    meas_std: np.ndarray    # (14,) cm
    hw_mean: np.ndarray     # (2,)  [height_cm, weight_kg]
    hw_std: np.ndarray      # (2,)

    def norm_meas(self, m: np.ndarray) -> np.ndarray:
        return (m - self.meas_mean) / self.meas_std

    def denorm_meas(self, z: np.ndarray) -> np.ndarray:
        return z * self.meas_std + self.meas_mean

    def norm_hw(self, hw: np.ndarray) -> np.ndarray:
        return (hw - self.hw_mean) / self.hw_std

    def to_dict(self) -> dict:
        return {
            "meas_mean": self.meas_mean, "meas_std": self.meas_std,
            "hw_mean": self.hw_mean, "hw_std": self.hw_std,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        return cls(
            meas_mean=np.asarray(d["meas_mean"], np.float64),
            meas_std=np.asarray(d["meas_std"], np.float64),
            hw_mean=np.asarray(d["hw_mean"], np.float64),
            hw_std=np.asarray(d["hw_std"], np.float64),
        )
