"""BodyM split loader and the shared silhouette→tensor builder.

BodyM layout (per split ``train`` / ``testA`` / ``testB``)::

    hwg_metadata.csv          subject_id, gender, height_cm, weight_kg
    measurements.csv          subject_id, …14 measurements (cm)…
    subject_to_photo_map.csv  subject_id, photo_id
    mask/<photo_id>.png       frontal silhouette  (white body on black)
    mask_left/<photo_id>.png  lateral silhouette

One training sample is one ``photo_id`` (a subject may own several
photos in different clothing); the regression target is that subject's
measurement row. ``build_input`` is the single code path that turns two
silhouettes + height + weight into the 3-channel network tensor — used
identically by training and by inference on captured photos.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import MEAS_COLS
from .model import Standardizer


def build_input(front: np.ndarray, side: np.ndarray, height_cm: float,
                weight_kg: float, std: Standardizer, *,
                img_h: int, img_w: int) -> torch.Tensor:
    """Two silhouettes + H/W  →  a (3, img_h, 2*img_w) float tensor.

    ``front`` / ``side`` are 2-D arrays, any dtype; non-zero = body. Each
    is resized to ``(img_h, img_w)`` and laid side by side. Channels 1
    and 2 are constant planes holding the standardized height and weight,
    exactly as the paper concatenates H/W depthwise."""
    def prep(m: np.ndarray) -> np.ndarray:
        b = (np.asarray(m) > 0).astype(np.float32)
        return cv2.resize(b, (img_w, img_h), interpolation=cv2.INTER_AREA)

    sil = np.concatenate([prep(front), prep(side)], axis=1)   # (H, 2W)
    hwz = std.norm_hw(np.array([height_cm, weight_kg], np.float64))
    h_plane = np.full_like(sil, float(hwz[0]))
    w_plane = np.full_like(sil, float(hwz[1]))
    chw = np.stack([sil, h_plane, w_plane], axis=0)           # (3, H, 2W)
    return torch.from_numpy(chw.astype(np.float32))


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class BodyMDataset(Dataset):
    """One BodyM split as (input tensor, 14-measurement target) pairs."""

    def __init__(self, root: Path, split: str, std: Standardizer, *,
                 img_h: int = 256, img_w: int = 192) -> None:
        self.dir = Path(root) / split
        self.std = std
        self.img_h, self.img_w = img_h, img_w

        hwg = {r["subject_id"]: r
               for r in _read_csv(self.dir / "hwg_metadata.csv")}
        meas = {r["subject_id"]: r
                for r in _read_csv(self.dir / "measurements.csv")}

        self.samples: list[tuple[str, str]] = []   # (photo_id, subject_id)
        for r in _read_csv(self.dir / "subject_to_photo_map.csv"):
            sid, pid = r["subject_id"], r["photo_id"]
            if sid not in hwg or sid not in meas:
                continue
            if not (self.dir / "mask" / f"{pid}.png").exists():
                continue
            if not (self.dir / "mask_left" / f"{pid}.png").exists():
                continue
            self.samples.append((pid, sid))

        self.hwg, self.meas = hwg, meas

    def __len__(self) -> int:
        return len(self.samples)

    def raw_targets(self) -> np.ndarray:
        """(N, 14) measurement matrix in cm — for computing statistics."""
        return np.array(
            [[float(self.meas[sid][c]) for c in MEAS_COLS]
             for _, sid in self.samples], np.float64)

    def raw_hw(self) -> np.ndarray:
        """(N, 2) [height_cm, weight_kg] — for computing statistics."""
        return np.array(
            [[float(self.hwg[sid]["height_cm"]),
              float(self.hwg[sid]["weight_kg"])]
             for _, sid in self.samples], np.float64)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        pid, sid = self.samples[i]
        front = cv2.imread(str(self.dir / "mask" / f"{pid}.png"),
                           cv2.IMREAD_GRAYSCALE)
        side = cv2.imread(str(self.dir / "mask_left" / f"{pid}.png"),
                          cv2.IMREAD_GRAYSCALE)
        h = float(self.hwg[sid]["height_cm"])
        w = float(self.hwg[sid]["weight_kg"])
        x = build_input(front, side, h, w, self.std,
                        img_h=self.img_h, img_w=self.img_w)
        m = np.array([float(self.meas[sid][c]) for c in MEAS_COLS],
                     np.float64)
        z = self.std.norm_meas(m).astype(np.float32)
        return x, torch.from_numpy(z)


def compute_standardizer(root: Path, split: str = "train") -> Standardizer:
    """Fit measurement + H/W mean/std on a split (use ``train``)."""
    ds = BodyMDataset(root, split,
                      Standardizer(np.zeros(14), np.ones(14),
                                   np.zeros(2), np.ones(2)))
    t, hw = ds.raw_targets(), ds.raw_hw()
    return Standardizer(
        meas_mean=t.mean(0), meas_std=t.std(0) + 1e-6,
        hw_mean=hw.mean(0), hw_std=hw.std(0) + 1e-6,
    )
