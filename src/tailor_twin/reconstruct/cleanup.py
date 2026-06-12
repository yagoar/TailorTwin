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
    verbose: bool = True,
) -> o3d.geometry.TriangleMesh:
    """Run the full cleanup pipeline (component → fill → smooth → decimate)."""
    def _shape(m: o3d.geometry.TriangleMesh) -> str:
        return f"{len(m.vertices)} v / {len(m.triangles)} f"

    if verbose:
        print(f"  cleanup in:    {_shape(mesh)}")
    m = keep_largest_component(mesh)
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
