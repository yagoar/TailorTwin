"""Post-TSDF mesh cleanup.

TSDF fusion tends to produce:
  - floating fragments where transient noise crossed the volume
  - small holes at glancing-angle surfaces
  - per-voxel staircase noise on the surface

This module trims fragments, fills small holes, and smooths the
surface enough for SMPL-X fitting without sanding away anatomical
detail (bust apex, scapular ridge).
"""
from __future__ import annotations

import numpy as np
import open3d as o3d


# Smoothing iterations: 5 is the SPEC-aligned default — enough to kill
# voxel staircase, not enough to blur landmarks.
DEFAULT_SMOOTH_ITERS = 5

# Hole-fill cap in metres² — fills small gaps left by depth noise but
# leaves intentional concavities (armpit, between fingers) untouched.
DEFAULT_HOLE_FILL_AREA_M2 = 0.003   # ~3 cm²

# Triangle-count target for the final mesh. SMPL-X fitting reads scan
# vertices into a tensor; cutting the scan to ~80 k tris keeps the
# Chamfer KDTree fast without losing measurement-grade detail.
DEFAULT_TARGET_TRIS = 80_000


def remove_floor_plane(
    mesh: o3d.geometry.TriangleMesh,
    *,
    dist_thresh_m: float = 0.012,
    band_frac: float = 0.18,
    min_inlier_frac: float = 0.04,
) -> o3d.geometry.TriangleMesh:
    """Detect and remove a horizontal floor patch fused under the feet.

    Even with body segmentation, depth pixels of the floor the subject
    stands on leak into the fuse. When that patch connects to the feet it
    survives ``keep_largest_component`` and inflates the mesh height —
    which corrupts every height-proportional landmark Y downstream (a fit
    that stretches to reach the floor reads a wrong stature, and the waist/
    high-hip slices then cut the body at the wrong level).

    Strategy: RANSAC a plane on the vertices in the bottom ``band_frac`` of
    the mesh height. If the dominant plane there is near-horizontal (normal
    within ~25 deg of vertical) and captures a meaningful share of those
    points, its inliers are floor — remove them. Conservative by design:
    no horizontal plane found (e.g. clean feet only) → mesh untouched.
    """
    V = np.asarray(mesh.vertices)
    if len(V) < 100:
        return mesh
    y = V[:, 1]
    y_lo, y_hi = float(y.min()), float(y.max())
    band_top = y_lo + band_frac * (y_hi - y_lo)
    band_idx = np.where(y <= band_top)[0]
    if band_idx.size < 50:
        return mesh

    band_pcd = o3d.geometry.PointCloud()
    band_pcd.points = o3d.utility.Vector3dVector(V[band_idx])
    try:
        plane, inl = band_pcd.segment_plane(
            distance_threshold=dist_thresh_m, ransac_n=3, num_iterations=500)
    except Exception:  # noqa: BLE001 — degenerate band
        return mesh
    if len(inl) < min_inlier_frac * len(V):
        return mesh
    # Horizontal plane => |normal_y| ~ 1. cos(25deg) ~ 0.906.
    if abs(plane[1]) < 0.906:
        return mesh

    floor_vert_mask = np.zeros(len(V), dtype=bool)
    floor_vert_mask[band_idx[np.asarray(inl)]] = True
    out = o3d.geometry.TriangleMesh(mesh)
    out.remove_vertices_by_mask(floor_vert_mask)
    out.remove_unreferenced_vertices()
    return out


def keep_largest_component(
    mesh: o3d.geometry.TriangleMesh,
) -> o3d.geometry.TriangleMesh:
    """Drop everything except the largest connected mesh component.

    Removes the helper's hand / floor patches / stray TSDF crumbs that
    survived segmentation. Returns a NEW mesh (the input is not mutated).
    """
    out = o3d.geometry.TriangleMesh(mesh)
    triangle_clusters, _cluster_n_tris, _cluster_area = (
        out.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    if triangle_clusters.size == 0:
        return out
    sizes = np.bincount(triangle_clusters)
    biggest = int(np.argmax(sizes))
    keep_tri = triangle_clusters == biggest
    out.remove_triangles_by_mask(~keep_tri)
    out.remove_unreferenced_vertices()
    return out


def fill_small_holes(
    mesh: o3d.geometry.TriangleMesh,
    max_hole_area_m2: float = DEFAULT_HOLE_FILL_AREA_M2,
) -> o3d.geometry.TriangleMesh:
    """Close small holes left by depth noise.

    Open3D's `fill_holes()` was added in 0.16 on the t.geometry side.
    We convert via Tensor mesh, fill, then convert back to legacy.
    Holes whose triangulated-fan area exceeds the cap are left open
    (we don't want to bridge real concavities like armpits).
    """
    if not hasattr(o3d.t.geometry.TriangleMesh, "from_legacy"):
        return mesh  # very old Open3D — skip
    tm = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    filled = tm.fill_holes(hole_size=float(max_hole_area_m2))
    return filled.to_legacy()


def smooth_laplacian(
    mesh: o3d.geometry.TriangleMesh,
    iters: int = DEFAULT_SMOOTH_ITERS,
) -> o3d.geometry.TriangleMesh:
    """Per-vertex Laplacian smooth — Taubin-flavoured to limit shrinkage.

    Open3D's `filter_smooth_taubin` alternates +λ / -μ passes so the
    mesh doesn't collapse inward. 5 iters at default λ/μ is a SPEC
    tailor-twin balance.
    """
    if iters <= 0:
        return mesh
    return mesh.filter_smooth_taubin(number_of_iterations=iters)


def decimate(
    mesh: o3d.geometry.TriangleMesh,
    target_tris: int = DEFAULT_TARGET_TRIS,
) -> o3d.geometry.TriangleMesh:
    """Decimate to ``target_tris`` quadric-error-style.

    No-op if the mesh is already smaller than target. SMPL-X fitting
    KDTree builds are O(N log N); cutting from ~300 k → 80 k tris
    speeds the Chamfer loss by ~3-4×.
    """
    if len(mesh.triangles) <= target_tris:
        return mesh
    return mesh.simplify_quadric_decimation(target_number_of_triangles=target_tris)


def rescale_to_stature(
    mesh: o3d.geometry.TriangleMesh,
    height_m: float,
    *,
    up_axis: int = 1,
    verbose: bool = True,
) -> tuple[o3d.geometry.TriangleMesh, float]:
    """Uniformly rescale ``mesh`` so its vertical extent equals ``height_m``.

    Absolute metric scale comes from the LiDAR depth, but visual-inertial
    odometry drift over a multi-loop capture leaves a small *global* scale
    error baked into the fused mesh. One tape-measured standing height is a
    hard anchor that removes it: we scale the whole mesh isotropically so the
    floor→crown extent along the gravity-up axis matches the real number.

    Isotropic (not Y-only) on purpose — odometry scale error is global, so a
    single uniform factor corrects it without distorting girths. A Y-only
    scale would stretch heights while leaving widths wrong.

    ARKit's world frame is gravity-aligned with +Y up, so ``up_axis=1`` is the
    stature axis. The mesh must cover crown→floor: if the capture missed the
    head or feet the measured extent is short and the factor is wrong, so this
    warns when the correction is implausibly large.

    Returns ``(scaled_mesh, factor)``. ``factor`` near 1.0 means the raw fuse
    was already well-scaled (good odometry); a large deviation flags drift.
    """
    v = np.asarray(mesh.vertices)
    if v.size == 0:
        if verbose:
            print("  rescale: empty mesh, skipped")
        return mesh, 1.0
    extent = float(v[:, up_axis].max() - v[:, up_axis].min())
    if extent <= 1e-6:
        if verbose:
            print(f"  rescale: degenerate up-axis extent {extent:.4f} m, skipped")
        return mesh, 1.0
    factor = float(height_m) / extent
    if not (0.5 < factor < 2.0) and verbose:
        print(f"  rescale: WARNING factor={factor:.3f} (mesh extent "
              f"{extent*100:.1f} cm vs target {height_m*100:.1f} cm) — "
              "capture may be missing the crown or feet; height anchor "
              "unreliable")
    # Uniform scale about the floor (min up-axis) so the feet stay grounded.
    center = v.min(axis=0).astype(np.float64)
    out = o3d.geometry.TriangleMesh(mesh)
    out.scale(factor, center=center)
    out.compute_vertex_normals()
    if verbose:
        print(f"  rescale:       extent {extent*100:.1f} cm → "
              f"{height_m*100:.1f} cm (factor {factor:.4f})")
    return out, factor


def cleanup_mesh(
    mesh: o3d.geometry.TriangleMesh,
    *,
    smooth_iters: int = DEFAULT_SMOOTH_ITERS,
    fill_hole_area_m2: float = DEFAULT_HOLE_FILL_AREA_M2,
    target_tris: int = DEFAULT_TARGET_TRIS,
    remove_floor: bool = True,
    verbose: bool = True,
) -> o3d.geometry.TriangleMesh:
    """Run the full cleanup pipeline (floor → component → fill → smooth → decimate)."""
    def _shape(m: o3d.geometry.TriangleMesh) -> str:
        return f"{len(m.vertices)} v / {len(m.triangles)} f"

    if verbose:
        print(f"  cleanup in:    {_shape(mesh)}")
    m = mesh
    if remove_floor:
        m = remove_floor_plane(m)
        if verbose:
            print(f"  remove-floor:  {_shape(m)}")
    m = keep_largest_component(m)
    if verbose:
        print(f"  keep-largest:  {_shape(m)}")
    m = fill_small_holes(m, max_hole_area_m2=fill_hole_area_m2)
    if verbose:
        print(f"  fill-holes:    {_shape(m)}")
    m = smooth_laplacian(m, iters=smooth_iters)
    if verbose:
        print(f"  smooth:        {_shape(m)}")
    m = decimate(m, target_tris=target_tris)
    if verbose:
        print(f"  decimate:      {_shape(m)}")
    m.compute_vertex_normals()
    return m
