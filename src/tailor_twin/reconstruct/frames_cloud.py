"""Multi-frame point cloud — the fusion-free chamfer target (EXPERIMENTAL).

Why this exists (ROADMAP Workstream B1): TSDF fusion averages every frame
into one surface, so subject sway over the 30-45 s orbit and odometry
drift are baked into the mesh as fuzz/doubling that caps the fit quality
— and the fused mesh is only ever used as a chamfer target anyway (the
measured body is always SMPL-X+D). The SMPL-X fit consumes points, not a
surface. This module back-projects the segmented, confidence-filtered
depth of the kept frames straight into one world-frame point cloud,
skipping fusion entirely: sway becomes per-frame point scatter that the
fit averages in the LOSS instead of baking into geometry.

Conventions: the back-projection mirrors ``reconstruct/tsdf.py`` exactly
— Open3D/OpenCV pinhole (x=(u-cx)·z/fx, y=(v-cy)·z/fy, z=+depth) and
``world = pose_c2w @ cam`` — so the cloud lands in the SAME world frame
as the fused meshes the pipeline is validated on. First-run sanity check
on a real capture: overlay ``*_scan_cloud.obj`` with a previous
``*_scan.obj`` from the same capture in MeshLab — they must coincide.

Pure numpy + scipy (no Open3D import) so every function here is
unit-testable without the ML stack.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_DEPTH_SCALE = 1000.0   # Stray depth_mm → metres (mirrors tsdf.py)
DEFAULT_DEPTH_TRUNC_M = 3.0    # drop returns beyond this (mirrors tsdf.py)
DEFAULT_VOXEL_M = 0.005        # merge grid; body surface ≈ 1.8 m² → ~70 k pts


def rescale_intrinsics(
    K: np.ndarray,
    native_size: tuple[int, int] | None,
    depth_size: tuple[int, int],
) -> np.ndarray:
    """fx/fy/cx/cy from the RGB-native pixel grid into the depth grid.

    Mirrors ``tsdf._maybe_rescale_intrinsics`` (kept separate so this
    module stays importable without Open3D). ``native_size`` /
    ``depth_size`` are (width, height).
    """
    if native_size is None or native_size == depth_size:
        return K
    nw, nh = native_size
    dw, dh = depth_size
    Kp = K.copy().astype(np.float64)
    Kp[0, 0] *= dw / nw
    Kp[1, 1] *= dh / nh
    Kp[0, 2] *= dw / nw
    Kp[1, 2] *= dh / nh
    return Kp


def backproject_depth(
    depth_mm: np.ndarray,
    K_depth_grid: np.ndarray,
    pose_c2w: np.ndarray,
    *,
    depth_scale: float = DEFAULT_DEPTH_SCALE,
    depth_trunc_m: float = DEFAULT_DEPTH_TRUNC_M,
) -> np.ndarray:
    """Lift one masked depth image to world-frame points, (N, 3) float64.

    Zero depth (masked-out pixels) and returns beyond ``depth_trunc_m``
    are dropped. Projection convention matches what Open3D applies inside
    ``tsdf.fuse_frames`` (extrinsic = inv(pose_c2w), OpenCV pinhole), so
    for identical inputs the points coincide with the TSDF surface.
    """
    h, w = depth_mm.shape
    z = depth_mm.astype(np.float64) / depth_scale
    valid = (z > 0.0) & (z <= depth_trunc_m)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float64)
    vs, us = np.nonzero(valid)
    zz = z[vs, us]
    fx, fy = K_depth_grid[0, 0], K_depth_grid[1, 1]
    cx, cy = K_depth_grid[0, 2], K_depth_grid[1, 2]
    xc = (us.astype(np.float64) - cx) * zz / fx
    yc = (vs.astype(np.float64) - cy) * zz / fy
    pts_cam = np.stack([xc, yc, zz, np.ones_like(zz)], axis=1)
    pts_world = pts_cam @ pose_c2w.astype(np.float64).T
    return pts_world[:, :3]


def grid_downsample(points: np.ndarray, voxel_m: float) -> np.ndarray:
    """Average points within each ``voxel_m`` grid cell (≈ Open3D's
    ``voxel_down_sample``, numpy-only). ``voxel_m <= 0`` returns input."""
    if voxel_m <= 0 or len(points) == 0:
        return points
    keys = np.floor(points / voxel_m).astype(np.int64)
    # Unique voxel per point → mean of members. np.unique gives an inverse
    # index we can accumulate with bincount per axis.
    _, inv, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True)
    out = np.zeros((len(counts), 3), dtype=np.float64)
    for axis in range(3):
        out[:, axis] = np.bincount(inv, weights=points[:, axis]) / counts
    return out


def remove_statistical_outliers(
    points: np.ndarray,
    *,
    k: int = 8,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """Drop points whose mean distance to their ``k`` nearest neighbours
    is more than ``std_ratio`` standard deviations above the global mean
    (≈ Open3D's ``remove_statistical_outlier``). Catches segmentation
    spill (matte halo pixels) that back-projects as floating specks a
    chamfer target must not contain."""
    n = len(points)
    if k <= 0 or n <= k + 1:
        return points
    from scipy.spatial import cKDTree

    d, _ = cKDTree(points).query(points, k=k + 1)  # col 0 is self (d=0)
    mean_d = d[:, 1:].mean(axis=1)
    keep = mean_d <= mean_d.mean() + std_ratio * mean_d.std()
    return points[keep]


def rescale_points_to_stature(
    points: np.ndarray,
    height_m: float,
    *,
    up_axis: int = 1,
    verbose: bool = True,
) -> tuple[np.ndarray, float]:
    """Uniformly rescale the cloud so its vertical extent equals
    ``height_m`` — same semantics as ``cleanup.rescale_to_stature``
    (isotropic, about the min corner so the feet stay grounded)."""
    if len(points) == 0:
        return points, 1.0
    extent = float(points[:, up_axis].max() - points[:, up_axis].min())
    if extent <= 1e-6:
        if verbose:
            print(f"  rescale: degenerate extent {extent:.4f} m, skipped")
        return points, 1.0
    factor = float(height_m) / extent
    if not (0.5 < factor < 2.0) and verbose:
        print(f"  rescale: WARNING factor={factor:.3f} (cloud extent "
              f"{extent*100:.1f} cm vs target {height_m*100:.1f} cm) — "
              "capture may be missing the crown or feet")
    corner = points.min(axis=0)
    out = corner + (points - corner) * factor
    if verbose:
        print(f"  rescale:       extent {extent*100:.1f} cm → "
              f"{height_m*100:.1f} cm (factor {factor:.4f})")
    return out, factor


def build_frames_cloud(
    inputs: Iterable,
    *,
    intrinsics_native_size: tuple[int, int] | None = None,
    voxel_m: float = DEFAULT_VOXEL_M,
    depth_scale: float = DEFAULT_DEPTH_SCALE,
    depth_trunc_m: float = DEFAULT_DEPTH_TRUNC_M,
    outlier_k: int = 8,
    outlier_std_ratio: float = 2.0,
    progress: bool = True,
) -> np.ndarray:
    """Back-project every input frame and merge into one cleaned cloud.

    ``inputs`` are ``tsdf.FusionInput``-shaped records (attributes
    ``depth_mm``, ``intrinsics``, ``pose_c2w``) — duck-typed so this
    module never imports Open3D. Each frame is grid-downsampled before
    merging (bounds memory), the merged cloud is downsampled again (one
    representative point per voxel across ALL frames — this is where
    per-frame sway scatter gets averaged), then statistical outliers are
    dropped.
    """
    per_frame: list[np.ndarray] = []
    count = 0
    for fi in inputs:
        h, w = fi.depth_mm.shape
        K = rescale_intrinsics(fi.intrinsics, intrinsics_native_size, (w, h))
        pts = backproject_depth(
            fi.depth_mm, K, fi.pose_c2w,
            depth_scale=depth_scale, depth_trunc_m=depth_trunc_m)
        if len(pts):
            per_frame.append(grid_downsample(pts, voxel_m))
        count += 1
        if progress and count % 50 == 0:
            print(f"  cloud: back-projected {count} frames")
    if not per_frame:
        raise RuntimeError("build_frames_cloud: no points from any frame")
    merged = np.concatenate(per_frame, axis=0)
    cloud = grid_downsample(merged, voxel_m)
    cleaned = remove_statistical_outliers(
        cloud, k=outlier_k, std_ratio=outlier_std_ratio)
    if progress:
        print(f"  cloud: {count} frames → {len(merged)} pts → "
              f"{len(cloud)} merged → {len(cleaned)} after outlier removal")
    return cleaned


def save_cloud_obj(points: np.ndarray, path: Path) -> None:
    """Write a points-only Wavefront OBJ (``v`` lines, no faces). Opens in
    MeshLab/Blender for the overlay sanity check and round-trips through
    :func:`load_cloud_obj`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# TailorTwin multi-frame point cloud (fusion-free)\n")
        for x, y, z in np.asarray(points, dtype=np.float64):
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")


def load_cloud_obj(path: Path) -> np.ndarray:
    """Read the ``v`` lines of an OBJ into (N, 3) float32. Tiny local
    parser so the cloud path does not depend on how trimesh models
    face-less OBJs."""
    pts: list[tuple[float, float, float]] = []
    with Path(path).open() as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                pts.append((float(x), float(y), float(z)))
    if not pts:
        raise ValueError(f"no vertices in {path}")
    return np.asarray(pts, dtype=np.float32)
