"""TSDF fusion of segmented + filtered Stray Scanner frames into a mesh.

Open3D's ``ScalableTSDFVolume`` integrates per-frame RGB-D + extrinsics.
Body-scale presets:

  voxel_length = 5 mm    # noise vs detail balance for a ~1.8 m subject
  sdf_trunc    = 20 mm   # 4× voxel_length per Open3D best-practice

We integrate world←camera extrinsics (Open3D convention) — Stray gives
camera→world (ARKit / right-handed +Y up), so we invert.

Confidence + depth-range filtering and body segmentation must be
applied to ``depth_mm`` *before* integration. This module does the
fusion only and assumes the inputs are already body-masked.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d


# Body-scale TSDF presets. Tunable from the CLI.
DEFAULT_VOXEL_M = 0.005           # 5 mm
DEFAULT_SDF_TRUNC_M = 0.020       # 4× voxel
DEFAULT_DEPTH_TRUNC_M = 3.0       # hard ceiling beyond which depth is dropped
DEFAULT_DEPTH_SCALE = 1000.0      # Stray depth_mm → metres


@dataclass
class FusionInput:
    """One frame's contribution to TSDF integration.

    ``depth_mm``    H×W uint16, mm, already segmentation-masked.
    ``intrinsics``  3×3, in the depth-image pixel grid.
    ``pose_c2w``    4×4 camera→world SE(3) (Stray's odometry convention).
    """
    depth_mm: np.ndarray
    intrinsics: np.ndarray
    pose_c2w: np.ndarray


def _intrinsics_to_o3d(K: np.ndarray, width: int, height: int) -> o3d.camera.PinholeCameraIntrinsic:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return o3d.camera.PinholeCameraIntrinsic(
        width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy,
    )


def _maybe_rescale_intrinsics(
    K: np.ndarray, native_size: tuple[int, int] | None,
    depth_size: tuple[int, int],
) -> np.ndarray:
    """Stray's odometry intrinsics may be in the RGB resolution. Open3D
    needs them in the same pixel grid as the depth image. If
    ``native_size`` is given and differs from the depth size, rescale
    fx/fy/cx/cy proportionally.

    Stray docs flag this as an open question (see io/stray_loader.py
    module docstring) — passing the rescale lets the caller resolve it
    once and forward consistent intrinsics here.
    """
    if native_size is None or native_size == depth_size:
        return K
    nw, nh = native_size
    dw, dh = depth_size
    sx = dw / nw
    sy = dh / nh
    Kp = K.copy().astype(np.float64)
    Kp[0, 0] *= sx        # fx
    Kp[1, 1] *= sy        # fy
    Kp[0, 2] *= sx        # cx
    Kp[1, 2] *= sy        # cy
    return Kp


def fuse_frames(
    inputs: Iterable[FusionInput],
    *,
    voxel_length: float = DEFAULT_VOXEL_M,
    sdf_trunc: float = DEFAULT_SDF_TRUNC_M,
    depth_trunc: float = DEFAULT_DEPTH_TRUNC_M,
    depth_scale: float = DEFAULT_DEPTH_SCALE,
    intrinsics_native_size: tuple[int, int] | None = None,
    progress: bool = True,
) -> o3d.geometry.TriangleMesh:
    """Integrate frames into a TSDF volume and return the extracted mesh.

    ``intrinsics_native_size`` (width, height) is the pixel grid in
    which Stray reports fx/fy/cx/cy. If None, the intrinsics are
    assumed to already match the depth resolution.
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )
    count = 0
    for fi in inputs:
        h, w = fi.depth_mm.shape
        K = _maybe_rescale_intrinsics(
            fi.intrinsics, intrinsics_native_size, (w, h))
        intr = _intrinsics_to_o3d(K, width=w, height=h)
        depth_o3d = o3d.geometry.Image(fi.depth_mm.astype(np.uint16))
        # Make a placeholder RGB the same size as depth; Open3D's
        # create_from_color_and_depth requires both even when fusing
        # without colour. Use a black single-channel image.
        rgb_dummy = o3d.geometry.Image(
            np.zeros((h, w, 3), dtype=np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb_dummy,
            depth=depth_o3d,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        # Open3D wants world→camera extrinsics; Stray gives camera→world.
        extr = np.linalg.inv(fi.pose_c2w)
        volume.integrate(rgbd, intr, extr)
        count += 1
        if progress and count % 50 == 0:
            print(f"  TSDF: integrated {count} frames")
    if count == 0:
        raise RuntimeError("fuse_frames: no inputs supplied")
    if progress:
        print(f"  TSDF: extracting mesh ({count} frames integrated)")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def _pcd_from_input(
    fi: FusionInput,
    *,
    depth_scale: float,
    depth_trunc: float,
    intrinsics_native_size: tuple[int, int] | None,
    voxel_down: float,
) -> o3d.geometry.PointCloud:
    """Back-project one masked depth frame to a camera-frame point cloud.

    Down-sampled + normal-estimated so point-to-plane ICP can run on it
    during pose-graph refinement.
    """
    h, w = fi.depth_mm.shape
    K = _maybe_rescale_intrinsics(fi.intrinsics, intrinsics_native_size, (w, h))
    intr = _intrinsics_to_o3d(K, width=w, height=h)
    depth_img = o3d.geometry.Image(fi.depth_mm.astype(np.uint16))
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_img, intr, depth_scale=depth_scale, depth_trunc=depth_trunc,
    )
    if voxel_down > 0:
        pcd = pcd.voxel_down_sample(voxel_down)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_down * 2.0, max_nn=30))
    return pcd


def _pairwise_register(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    *,
    max_corr: float,
    init: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Point-to-plane ICP from ``init``; return (transform, information, fitness)."""
    reg = o3d.pipelines.registration.registration_icp(
        source, target, max_corr, init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_corr, reg.transformation)
    return reg.transformation, info, float(reg.fitness)


def fuse_frames_posegraph(
    inputs: Iterable[FusionInput],
    *,
    voxel_length: float = DEFAULT_VOXEL_M,
    sdf_trunc: float = DEFAULT_SDF_TRUNC_M,
    depth_trunc: float = DEFAULT_DEPTH_TRUNC_M,
    depth_scale: float = DEFAULT_DEPTH_SCALE,
    intrinsics_native_size: tuple[int, int] | None = None,
    keyframe_stride: int = 1,
    loop_radius_m: float = 0.30,
    loop_min_gap: int = 10,
    loop_min_fitness: float = 0.30,
    progress: bool = True,
) -> o3d.geometry.TriangleMesh:
    """Drift-corrected TSDF fusion via Open3D multiway registration.

    EXPERIMENTAL — not yet validated against a real Stray capture. Default
    off; ``scan.py`` reaches it via ``--pose-graph``. The plain
    :func:`fuse_frames` (raw-odometry integration) remains the default path.

    Plain ``fuse_frames`` trusts the Stray odometry pose for every frame.
    ARKit visual-inertial odometry drifts over a multi-loop body capture, and
    naive integration *bakes that drift in* as doubled / ghosted surfaces that
    cannot be removed afterward. The standard fix (Open3D reconstruction
    system, Choi2015) refines the poses *before* integration:

      1. Back-project each keyframe to a point cloud (camera frame).
      2. Pre-place each by its odometry pose, then point-to-plane ICP between
         consecutive keyframes (odometry edges, ``uncertain=False``) and
         between spatially-near / temporally-distant keyframes that revisit
         the same surface (loop-closure edges, ``uncertain=True``).
      3. Global pose-graph optimization (Levenberg-Marquardt + line process)
         distributes the loop-closure constraints, cancelling accumulated
         drift and pruning false loops.
      4. TSDF-integrate the keyframes with the corrected extrinsics.

    ``keyframe_stride`` subsamples frames (pairwise ICP is O(n²); a few dozen
    keyframes is the sweet spot). Loop candidates are keyframe pairs within
    ``loop_radius_m`` in odometry translation but ≥ ``loop_min_gap`` apart in
    index; an ICP fitness below ``loop_min_fitness`` rejects the candidate.
    """
    frames = [fi for i, fi in enumerate(inputs) if i % max(keyframe_stride, 1) == 0]
    n = len(frames)
    if n == 0:
        raise RuntimeError("fuse_frames_posegraph: no inputs supplied")
    if n == 1:  # nothing to register — fall back to plain integration
        return fuse_frames(
            frames, voxel_length=voxel_length, sdf_trunc=sdf_trunc,
            depth_trunc=depth_trunc, depth_scale=depth_scale,
            intrinsics_native_size=intrinsics_native_size, progress=progress)

    if progress:
        print(f"  pose-graph: {n} keyframes (stride={keyframe_stride})")

    voxel_down = max(voxel_length * 2.0, 0.01)
    max_corr = voxel_down * 1.5

    # Camera-frame clouds, plus world-placed copies (by raw odometry) so ICP
    # starts from a near-identity init.
    pcds_cam = [
        _pcd_from_input(
            fi, depth_scale=depth_scale, depth_trunc=depth_trunc,
            intrinsics_native_size=intrinsics_native_size, voxel_down=voxel_down)
        for fi in frames
    ]
    poses = [fi.pose_c2w.astype(np.float64) for fi in frames]
    pcds_world = [o3d.geometry.PointCloud(p).transform(T)
                  for p, T in zip(pcds_cam, poses)]
    translations = np.stack([T[:3, 3] for T in poses])

    identity = np.eye(4)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(identity))
    odometry = identity.copy()
    n_loops = 0
    for src in range(n):
        for tgt in range(src + 1, n):
            sequential = tgt == src + 1
            if not sequential:
                gap_ok = (tgt - src) >= loop_min_gap
                near = float(np.linalg.norm(
                    translations[src] - translations[tgt])) <= loop_radius_m
                if not (gap_ok and near):
                    continue
            trans, info, fitness = _pairwise_register(
                pcds_world[src], pcds_world[tgt], max_corr=max_corr, init=identity)
            if sequential:
                odometry = trans @ odometry
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        src, tgt, trans, info, uncertain=False))
            else:
                if fitness < loop_min_fitness:
                    continue  # weak alignment — not a real revisit
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        src, tgt, trans, info, uncertain=True))
                n_loops += 1
    if progress:
        print(f"  pose-graph: {n_loops} loop-closure edge(s) accepted")

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_corr,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

    # Corrected camera→world: a camera point maps to world by the raw odometry
    # pose, then the optimized node transform lifts that world cloud into the
    # globally-consistent frame.
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )
    for idx, fi in enumerate(frames):
        h, w = fi.depth_mm.shape
        K = _maybe_rescale_intrinsics(
            fi.intrinsics, intrinsics_native_size, (w, h))
        intr = _intrinsics_to_o3d(K, width=w, height=h)
        depth_o3d = o3d.geometry.Image(fi.depth_mm.astype(np.uint16))
        rgb_dummy = o3d.geometry.Image(np.zeros((h, w, 3), dtype=np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb_dummy, depth=depth_o3d, depth_scale=depth_scale,
            depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
        corrected_c2w = pose_graph.nodes[idx].pose @ poses[idx]
        volume.integrate(rgbd, intr, np.linalg.inv(corrected_c2w))
    if progress:
        print(f"  pose-graph: integrating {n} corrected keyframes")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def save_mesh_obj(mesh: o3d.geometry.TriangleMesh, path: Path) -> None:
    """Write the fused mesh as Wavefront OBJ for the SMPL-X fit step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(path), mesh,
                                write_ascii=True, write_vertex_normals=True)
