"""Fuse multi-view Sapiens2 pointmap outputs into a single body point cloud.

Background
----------
Sapiens2's pointmap module produces, for each input image, a per-pixel
(x, y, z) prediction in the **camera coordinate frame**. Depth values
are scale-relative (not metric) and only the body's visible surface is
reliable; background pixels (walls, floor) carry large noisy depths.

When the input is a 360° rotation video of a *single* subject (subject
rotates, camera roughly stationary), each frame captures a different
slice of the same body. To recover a complete body cloud we:

  1. Filter every frame's pointmap to body-only points (optional seg mask).
  2. Map every frame from its camera-frame to a shared subject-centred
     frame. Initial guess comes from the known rotation pose of the
     subject in that frame; ICP refines the alignment using actual
     overlap between consecutive frames.
  3. Merge all aligned clouds and optionally scale so the final body
     height equals the user's tape height.

The resulting unified cloud is consumed by tailor-twin's existing
``fit.fit_scan`` (same chamfer-fit code that runs on the TSDF mesh),
so the downstream pipeline (measure CLI, viewer, garment) doesn't
change.

Implementation notes
--------------------
* Sapiens2 pointmaps have ~2 million points each → subsample aggressively
  before ICP, otherwise pairwise alignment is too slow.
* The first frame defines the reference orientation. All other frames
  align to it. The reference frame's yaw matters only for visual
  parity with downstream tools; you can rotate the merged cloud
  afterwards if you need a specific facing.
* Background filtering: if no segmentation masks are provided, we fall
  back to a depth-band heuristic — fit a 1-D Gaussian to the depth
  histogram and keep points within ±0.5 m of the dominant depth.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


# --------------------------------------------------------------------- #
# Frame description                                                     #
# --------------------------------------------------------------------- #


@dataclass
class PointmapFrame:
    """One Sapiens2-pointmap output prepared for fusion."""
    name: str                       # file stem, for logging
    points: np.ndarray              # (N, 3) float32 camera-frame
    colors: np.ndarray | None       # (N, 3) float32 in [0, 1], optional
    yaw_init_deg: float             # initial subject yaw guess for fusion


def _depth_band_mask(
    points: np.ndarray,
    band_m: float = 0.5,
) -> np.ndarray:
    """Keep points whose Z is within ±``band_m`` of the dominant depth.

    Sapiens2 puts background pixels at extreme negative Z (far / behind
    the camera plane); body pixels cluster around a single mode. We
    isolate that mode with a histogram peak instead of a fit because
    it's cheaper and the mode is always well separated."""
    z = points[:, 2]
    # Coarse 1-cm bins over the typical valid Z range. We only need to
    # find the dominant cluster — exact bin count isn't load bearing.
    hist, edges = np.histogram(z, bins=200, range=(-4.0, 2.0))
    peak_z = (edges[hist.argmax()] + edges[hist.argmax() + 1]) * 0.5
    return np.abs(z - peak_z) <= band_m


def _yaw_rotation(deg: float) -> np.ndarray:
    """4×4 SE(3) rotation around the +Y axis by ``deg`` degrees."""
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([[ c, 0,  s],
                          [ 0, 1,  0],
                          [-s, 0,  c]], dtype=np.float64)
    return T


def load_pointmap_ply(
    path: Path,
    yaw_init_deg: float,
    *,
    body_mask: np.ndarray | None = None,
    use_depth_band: bool = True,
    target_points: int = 80_000,
    rng_seed: int = 0,
) -> PointmapFrame:
    """Load one Sapiens2 .ply, filter to body-only, downsample, and
    return a ``PointmapFrame`` ready for fusion.

    ``body_mask`` (shape == pointmap H×W flattened) carries an external
    body-segmentation mask (e.g. from sapiens2-seg). If provided it's
    intersected with the depth-band heuristic; otherwise the depth band
    runs alone.
    """
    pc = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pc.points, dtype=np.float32)
    cols = (np.asarray(pc.colors, dtype=np.float32)
            if len(pc.colors) else None)

    keep = np.ones(len(pts), dtype=bool)
    if use_depth_band:
        keep &= _depth_band_mask(pts)
    if body_mask is not None:
        if body_mask.shape[0] != pts.shape[0]:
            raise ValueError(
                f"body_mask length {body_mask.shape[0]} != points "
                f"{pts.shape[0]} for {path}")
        keep &= body_mask
    pts = pts[keep]
    if cols is not None:
        cols = cols[keep]

    if target_points and len(pts) > target_points:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(pts), target_points, replace=False)
        pts = pts[idx]
        if cols is not None:
            cols = cols[idx]

    return PointmapFrame(
        name=path.stem,
        points=pts,
        colors=cols,
        yaw_init_deg=yaw_init_deg,
    )


# --------------------------------------------------------------------- #
# Alignment                                                             #
# --------------------------------------------------------------------- #


def _center(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return points centred at their median (robust to outliers) +
    the median itself so callers can undo the centring."""
    c = np.median(points, axis=0)
    return points - c, c


def _initial_transform(
    yaw_deg: float,
    centroid: np.ndarray,
) -> np.ndarray:
    """SE(3) that (a) translates the frame to its centroid origin
    and (b) rotates by ``-yaw_deg`` around +Y to align with frame 0."""
    # World frame puts the subject's centroid at origin and unrotates
    # the visible side back to "the frame 0 orientation". Order:
    # T_unrotate · T_centre.
    centre = np.eye(4)
    centre[:3, 3] = -centroid
    return _yaw_rotation(-yaw_deg) @ centre


def _to_open3d(pc_np: np.ndarray,
               colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pc_np.astype(np.float64))
    if colors is not None:
        pc.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pc


def _icp_refine(
    source_pc: o3d.geometry.PointCloud,
    target_pc: o3d.geometry.PointCloud,
    *,
    max_distance: float = 0.05,
    max_iter: int = 60,
) -> np.ndarray:
    """Point-to-point ICP. Returns the 4×4 refinement transform."""
    result = o3d.pipelines.registration.registration_icp(
        source_pc, target_pc, max_distance, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter, relative_fitness=1e-7,
            relative_rmse=1e-7,
        ),
    )
    return result.transformation


def fuse_frames(
    frames: list[PointmapFrame],
    *,
    refine_with_icp: bool = True,
    icp_max_distance: float = 0.08,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fuse a list of ``PointmapFrame`` into one (points, colors) tuple
    in a shared subject-centred frame.

    Frame 0 is the alignment reference. Each subsequent frame is
    centred at its median, rotated by its ``-yaw_init_deg``, then
    optionally ICP-refined onto the running merged cloud."""
    if not frames:
        raise ValueError("fuse_frames: need at least one frame")

    # Anchor frame: centre + rotate, no ICP needed.
    anchor = frames[0]
    anchor_pts, anchor_centroid = _center(anchor.points)
    anchor_T = _initial_transform(anchor.yaw_init_deg, anchor_centroid)
    anchor_pts_world = (anchor_T[:3, :3] @ anchor.points.T).T + anchor_T[:3, 3]
    merged_pts = anchor_pts_world.astype(np.float32)
    merged_cols = (anchor.colors.copy() if anchor.colors is not None else None)
    merged_pc = _to_open3d(merged_pts, merged_cols)

    for f in frames[1:]:
        # Initial guess.
        _, centroid = _center(f.points)
        T_init = _initial_transform(f.yaw_init_deg, centroid)
        pts_world = (T_init[:3, :3] @ f.points.T).T + T_init[:3, 3]

        if refine_with_icp:
            src_pc = _to_open3d(pts_world)
            T_refine = _icp_refine(
                src_pc, merged_pc, max_distance=icp_max_distance)
            pts_world = (T_refine[:3, :3] @ pts_world.T).T + T_refine[:3, 3]

        merged_pts = np.vstack([merged_pts, pts_world.astype(np.float32)])
        if merged_cols is not None and f.colors is not None:
            merged_cols = np.vstack([merged_cols, f.colors])
        elif f.colors is not None:
            # First-time appearance of colour data — back-fill prior None.
            merged_cols = np.vstack(
                [np.zeros((merged_pts.shape[0] - len(f.colors), 3),
                          dtype=np.float32),
                 f.colors])
        merged_pc = _to_open3d(merged_pts, merged_cols)
    return merged_pts, merged_cols


def metric_scale_by_height(
    points: np.ndarray,
    target_height_m: float,
) -> tuple[np.ndarray, float]:
    """Uniformly scale the cloud so its Y-extent (head→toe) equals
    ``target_height_m`` (e.g. user's tape height in metres).

    Returns the scaled points and the applied scale factor."""
    y = points[:, 1]
    # Use 1st/99th percentile to avoid stray outlier-driven scaling.
    span = float(np.percentile(y, 99) - np.percentile(y, 1))
    if span <= 0:
        raise ValueError("metric_scale_by_height: degenerate Y extent")
    scale = target_height_m / span
    return points * scale, scale


def save_fused(
    points: np.ndarray,
    colors: np.ndarray | None,
    out_ply: Path,
) -> None:
    out_ply = Path(out_ply)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    pc = _to_open3d(points, colors)
    o3d.io.write_point_cloud(str(out_ply), pc)


# --------------------------------------------------------------------- #
# Mesh wrapping (Poisson) — optional helper for visual inspection       #
# --------------------------------------------------------------------- #


def poisson_mesh(
    points: np.ndarray,
    out_obj: Path,
    *,
    depth: int = 10,
    normal_radius: float = 0.05,
    keep_quantile: float = 0.05,
) -> None:
    """Wrap a point cloud with Poisson surface reconstruction.

    Reasonable defaults for body scale (~1.7 m tall, ~80k points).
    Used for visual sanity-checking only — downstream chamfer fit
    consumes the point cloud directly."""
    pc = _to_open3d(points)
    pc.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=30))
    pc.orient_normals_consistent_tangent_plane(20)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=depth)
    # Trim low-density tails (extruded background fragments).
    densities = np.asarray(densities)
    keep = densities >= np.quantile(densities, keep_quantile)
    mesh.remove_vertices_by_mask(np.logical_not(keep))
    mesh.compute_vertex_normals()
    out_obj = Path(out_obj)
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_obj), mesh)
