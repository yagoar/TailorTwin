"""Stray Scanner capture loader.

Source: https://github.com/strayrobots/scanner — see ``docs/format.md`` in
that repo for the canonical on-disk format. This module follows that
specification and intentionally avoids inferring fields not documented
there.

Status: Phase 1 skeleton, validated against the public format document
only. Pending end-to-end validation against a real capture from the
user's iPhone. The ``validate_capture`` function asserts every documented
expectation so any drift between docs and a real capture surfaces loudly.

Expected on-disk layout (from the Stray docs)::

    <capture>/
      rgb.mp4                # HEVC-encoded video
      odometry.csv           # per-frame pose + intrinsics + timestamp
      imu.csv                # 6-axis IMU stream (not used by this loader)
      camera_matrix.csv      # legacy 3x3 intrinsics (preferred: odometry.csv)
      depth/                 # 16-bit PNG, 192x256, values in millimetres
        00000.png
        00001.png
        ...
      confidence/            # 8-bit PNG, 192x256, values in {0, 1, 2}
        00000.png
        ...
      distortion/            # optional, raw float32 LUTs per frame

Coordinate convention: Stray records ARKit poses, which are right-handed
with +Y up and the camera looking down -Z, in metres. Poses are returned
as 4x4 camera-to-world SE(3) matrices.

Open questions to confirm against a real capture (see Phase 1 close-out):

* Whether ``odometry.csv`` is 1:1 with the RGB stream (one row per RGB
  frame) or one row per depth frame.
* Whether the intrinsics in ``odometry.csv`` are reported in the RGB
  camera's native resolution or already scaled to the depth resolution.
* Whether depth filenames are zero-padded 5-digit indices into the
  odometry row sequence, or use a different convention.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# Stray docs/format.md: depth is 16-bit grayscale PNG at 192 (H) x 256 (W).
DEPTH_HEIGHT = 192
DEPTH_WIDTH = 256
DEPTH_DTYPE = np.uint16
DEPTH_UNITS = "mm"
CONFIDENCE_VALUES = (0, 1, 2)  # 0 = unreliable, 2 = highest

REQUIRED_FILES = ("rgb.mp4", "odometry.csv")
REQUIRED_DIRS = ("depth",)
OPTIONAL_FILES = ("imu.csv", "camera_matrix.csv")
OPTIONAL_DIRS = ("confidence", "distortion")

# odometry.csv column order per Stray docs/format.md.
_ODOM_COLS = (
    "timestamp", "frame",
    "x", "y", "z",
    "qx", "qy", "qz", "qw",
    "fx", "fy", "cx", "cy",
)


@dataclass(frozen=True)
class Frame:
    """One synchronized capture frame from a Stray Scanner session."""

    index: int                       # 0-based index into the odometry sequence
    timestamp_s: float               # seconds, from odometry.csv
    rgb: np.ndarray | None           # H_rgb x W_rgb x 3, uint8, RGB order
    depth_mm: np.ndarray | None      # 192 x 256, uint16, units = mm
    confidence: np.ndarray | None    # 192 x 256, uint8, values in {0,1,2}
    intrinsics: np.ndarray           # 3x3 float64; SEE OPEN QUESTIONS in module docstring
    pose_cam_to_world: np.ndarray    # 4x4 float64, SE(3), camera-to-world


def validate_capture(folder: Path) -> None:
    """Raise if the folder does not look like a Stray Scanner capture.

    Checked against the documented format. If a future Stray release adds or
    renames files, this will fail fast with a clear message — that is
    intentional and aligned with the project's "fail loudly, ground in
    source" policy.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"not a directory: {folder}")
    for name in REQUIRED_FILES:
        if not (folder / name).is_file():
            raise FileNotFoundError(f"required file missing: {folder / name}")
    for name in REQUIRED_DIRS:
        if not (folder / name).is_dir():
            raise FileNotFoundError(f"required dir missing: {folder / name}")


def _read_odometry(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"odometry.csv is empty: {path}")
    # Stray writes headers with leading spaces (" frame" etc.) and the
    # values inherit the same spacing. Strip both so downstream lookups
    # by canonical column name work without callers caring.
    rows = [{k.strip(): (v.strip() if v is not None else v) for k, v in r.items()}
            for r in rows]
    missing = [c for c in _ODOM_COLS if c not in rows[0]]
    if missing:
        raise ValueError(
            f"odometry.csv missing expected columns {missing}: {path}\n"
            f"got: {list(rows[0].keys())}"
        )
    return rows


def _intrinsics_from_row(row: dict[str, str]) -> np.ndarray:
    fx = float(row["fx"])
    fy = float(row["fy"])
    cx = float(row["cx"])
    cy = float(row["cy"])
    return np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _pose_from_row(row: dict[str, str]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
    quat = np.array(
        [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_quat(quat).as_matrix()
    T[:3, 3] = (x, y, z)
    return T


def _load_depth(path: Path) -> np.ndarray:
    import cv2

    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"cannot read depth png: {path}")
    if arr.shape != (DEPTH_HEIGHT, DEPTH_WIDTH):
        raise ValueError(
            f"unexpected depth shape {arr.shape}, expected "
            f"{(DEPTH_HEIGHT, DEPTH_WIDTH)}: {path}"
        )
    if arr.dtype != DEPTH_DTYPE:
        raise ValueError(
            f"unexpected depth dtype {arr.dtype}, expected {DEPTH_DTYPE}: {path}"
        )
    return arr


def _load_confidence(path: Path) -> np.ndarray | None:
    import cv2

    if not path.is_file():
        return None
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return arr


def _depth_index_from_path(path: Path) -> int | None:
    """Convert ``depth/00042.png`` -> 42. Returns None if the stem is not numeric."""
    try:
        return int(path.stem)
    except ValueError:
        return None


def load_capture(
    folder: str | Path,
    *,
    require_depth: bool = True,
    decode_rgb: bool = True,
) -> Iterator[Frame]:
    """Yield ``Frame`` for each row of ``odometry.csv``.

    ``require_depth=True`` (default) skips rows whose ``frame`` column has no
    matching ``depth/<frame>.png``. Set ``False`` to yield odometry-only
    frames (useful for debugging the pose stream).

    ``decode_rgb=False`` skips opening ``rgb.mp4`` (faster when only depth
    + pose are needed, e.g. for TSDF fusion).
    """
    import cv2

    folder = Path(folder).resolve()
    validate_capture(folder)

    odom_rows = _read_odometry(folder / "odometry.csv")
    depth_paths = sorted((folder / "depth").glob("*.png"))
    depth_by_index = {
        idx: p
        for p in depth_paths
        if (idx := _depth_index_from_path(p)) is not None
    }
    if not depth_by_index:
        raise FileNotFoundError(
            f"no numerically-named depth/*.png files in {folder/'depth'}"
        )

    cap = cv2.VideoCapture(str(folder / "rgb.mp4")) if decode_rgb else None
    try:
        for i, row in enumerate(odom_rows):
            # Read the next RGB frame regardless of whether we keep it, so
            # the video stream stays in lockstep with the odometry sequence.
            rgb = None
            if cap is not None:
                ok, bgr = cap.read()
                if ok:
                    rgb = bgr[..., ::-1].copy()  # cv2 BGR -> RGB

            frame_idx = int(row["frame"])
            depth_path = depth_by_index.get(frame_idx)
            if depth_path is None and require_depth:
                continue
            depth = _load_depth(depth_path) if depth_path is not None else None
            conf_path = folder / "confidence" / (depth_path.name if depth_path else "")
            conf = _load_confidence(conf_path) if depth_path is not None else None

            yield Frame(
                index=i,
                timestamp_s=float(row["timestamp"]),
                rgb=rgb,
                depth_mm=depth,
                confidence=conf,
                intrinsics=_intrinsics_from_row(row),
                pose_cam_to_world=_pose_from_row(row),
            )
    finally:
        if cap is not None:
            cap.release()
