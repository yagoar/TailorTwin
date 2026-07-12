"""Unit tests for the fusion-free multi-frame point cloud (ROADMAP B1).

Everything here is pure numpy/scipy — the module must stay importable and
testable without Open3D/torch. The projection convention itself (OpenCV
pinhole + pose_c2w, identical to what Open3D applies inside
``tsdf.fuse_frames``) is asserted analytically here; the world-frame
agreement with a real fused mesh is a user-machine check (overlay
``*_scan_cloud.obj`` with ``*_scan.obj`` in MeshLab).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tailor_twin.reconstruct.frames_cloud import (
    backproject_depth,
    build_frames_cloud,
    grid_downsample,
    load_cloud_obj,
    remove_statistical_outliers,
    rescale_intrinsics,
    rescale_points_to_stature,
    save_cloud_obj,
)


def _K(fx=200.0, fy=210.0, cx=128.0, cy=96.0) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def test_backproject_principal_point_identity_pose() -> None:
    depth = np.zeros((192, 256), dtype=np.uint16)
    depth[96, 128] = 1500  # 1.5 m at the principal point
    pts = backproject_depth(depth, _K(), np.eye(4))
    assert pts.shape == (1, 3)
    # OpenCV pinhole: the principal-point ray is the +Z axis, no Y flip.
    np.testing.assert_allclose(pts[0], [0.0, 0.0, 1.5], atol=1e-9)


def test_backproject_offaxis_pixel_matches_pinhole_model() -> None:
    K = _K()
    depth = np.zeros((192, 256), dtype=np.uint16)
    u, v, z_mm = 178, 40, 2000
    depth[v, u] = z_mm
    pts = backproject_depth(depth, K, np.eye(4))
    z = z_mm / 1000.0
    expect = [(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z]
    np.testing.assert_allclose(pts[0], expect, atol=1e-9)


def test_backproject_applies_pose_c2w() -> None:
    depth = np.zeros((192, 256), dtype=np.uint16)
    depth[96, 128] = 1000  # camera-frame (0, 0, 1)
    pose = np.eye(4)
    # Rotate 90° about Y (camera +Z → world +X) and translate.
    pose[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    pose[:3, 3] = (10.0, 20.0, 30.0)
    pts = backproject_depth(depth, _K(), pose)
    np.testing.assert_allclose(pts[0], [11.0, 20.0, 30.0], atol=1e-9)


def test_backproject_drops_zero_and_truncated_depth() -> None:
    depth = np.zeros((192, 256), dtype=np.uint16)
    depth[10, 10] = 500       # keep
    depth[20, 20] = 3500      # beyond 3 m trunc — drop
    pts = backproject_depth(depth, _K(), np.eye(4))
    assert len(pts) == 1


def test_rescale_intrinsics_matches_tsdf_semantics() -> None:
    K = _K(fx=1500.0, fy=1500.0, cx=960.0, cy=720.0)  # RGB-native 1920x1440
    Kd = rescale_intrinsics(K, (1920, 1440), (256, 192))
    assert Kd[0, 0] == pytest.approx(1500.0 * 256 / 1920)
    assert Kd[1, 1] == pytest.approx(1500.0 * 192 / 1440)
    assert Kd[0, 2] == pytest.approx(960.0 * 256 / 1920)
    assert Kd[1, 2] == pytest.approx(720.0 * 192 / 1440)
    # None or matching size → unchanged object semantics.
    np.testing.assert_array_equal(rescale_intrinsics(K, None, (256, 192)), K)


def test_grid_downsample_merges_within_voxel() -> None:
    pts = np.array([[0.001, 0.001, 0.001],
                    [0.003, 0.003, 0.003],   # same 5 mm voxel as above
                    [0.100, 0.100, 0.100]])
    out = grid_downsample(pts, 0.005)
    assert len(out) == 2
    merged = out[np.argmin(out[:, 0])]
    np.testing.assert_allclose(merged, [0.002, 0.002, 0.002], atol=1e-12)
    # voxel <= 0 → passthrough.
    assert grid_downsample(pts, 0.0) is pts


def test_outlier_removal_drops_floating_speck() -> None:
    rng = np.random.default_rng(0)
    body = rng.normal(scale=0.05, size=(500, 3))    # dense blob
    speck = np.array([[5.0, 5.0, 5.0]])              # far-away matte halo
    cleaned = remove_statistical_outliers(np.vstack([body, speck]), k=8)
    assert len(cleaned) <= 500
    assert not (np.abs(cleaned - 5.0) < 0.5).all(axis=1).any()


def test_rescale_points_to_stature() -> None:
    pts = np.array([[0.0, 0.0, 0.0], [0.5, 2.0, 0.5]])
    out, factor = rescale_points_to_stature(pts, 1.6, verbose=False)
    assert factor == pytest.approx(0.8)
    assert out[:, 1].max() - out[:, 1].min() == pytest.approx(1.6)
    np.testing.assert_allclose(out[0], [0.0, 0.0, 0.0])  # min corner pinned


def test_build_frames_cloud_two_frames_and_native_rescale() -> None:
    # Both frames report intrinsics in a 2x RGB-native grid (as Stray
    # does); one native size applies to the whole capture. Principal
    # point in the native grid is (256, 192) → (128, 96) in the depth
    # grid after rescale, so the marked pixel is the optical axis.
    K_native = _K(400.0, 420.0, 256.0, 192.0)
    depth_a = np.zeros((192, 256), dtype=np.uint16)
    depth_a[96, 128] = 1000
    depth_b = depth_a.copy()
    pose_b = np.eye(4)
    pose_b[:3, 3] = (1.0, 0.0, 0.0)
    frames = [
        SimpleNamespace(depth_mm=depth_a, intrinsics=K_native,
                        pose_c2w=np.eye(4)),
        SimpleNamespace(depth_mm=depth_b, intrinsics=K_native,
                        pose_c2w=pose_b),
    ]
    pts = build_frames_cloud(frames, intrinsics_native_size=(512, 384),
                             progress=False, outlier_k=0)
    assert len(pts) == 2
    xs = sorted(pts[:, 0])
    assert xs[0] == pytest.approx(0.0, abs=1e-9)
    assert xs[1] == pytest.approx(1.0, abs=1e-9)


def test_build_frames_cloud_raises_on_empty() -> None:
    frames = [SimpleNamespace(depth_mm=np.zeros((4, 4), dtype=np.uint16),
                              intrinsics=_K(), pose_c2w=np.eye(4))]
    with pytest.raises(RuntimeError):
        build_frames_cloud(frames, progress=False)


def test_cloud_obj_roundtrip(tmp_path: Path) -> None:
    pts = np.array([[0.1, -1.25, 0.33], [2.0, 0.0, -0.5]])
    path = tmp_path / "cloud.obj"
    save_cloud_obj(pts, path)
    back = load_cloud_obj(path)
    np.testing.assert_allclose(back, pts, atol=1e-6)
    with pytest.raises(ValueError):
        empty = tmp_path / "empty.obj"
        empty.write_text("# nothing\n")
        load_cloud_obj(empty)
