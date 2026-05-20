"""Ring deformation — hit tape-measured circumferences exactly.

Background
----------
A parametric body fit (SMPL-X betas, or any learned shape space) cannot
satisfy an arbitrary set of girth targets simultaneously: the shape
space couples correlated dimensions (e.g. highbust and bust are driven
by the same betas, so you cannot freely set highbust 4 cm below bust).
The chamfer fit gets the body 95 % right; the last few centimetres on
each girth need a *geometric* edit, not another beta solve.

Ring deformation does exactly that. For each target circumference it:

  1. Slices the mesh at the measurement's anatomical Y level.
  2. Measures the current circumference (same convex-hull-perimeter
     method as ``measure.primitives.PlanarGirth``).
  3. Computes a radial scale = target / current.
  4. Scales every mesh vertex in a Y-band around that level outward /
     inward from the slice centroid, with a smooth cosine falloff so
     neighbouring rings blend rather than step.

Because each ring is scaled independently, there is no shape-space
coupling — every targeted girth lands on its target. Topology is
untouched (only vertex positions move), so the SMPL-X measurement
extractor keeps working on the result.

Limitations
-----------
* Radial scaling is uniform around the ring — it changes girth without
  changing the front/back *ratio*. If a target needs an asymmetric
  cross-section change, ring deformation alone won't do it (that needs
  per-vertex displacement driven by real depth data).
* Only horizontal-plane circumferences (PlanarGirth-style codes) are
  supported. Geodesic / TapeLoop codes (e.g. G03 highbust) are not
  directly deformable here — but they sit adjacent to a PlanarGirth
  ring and move sympathetically when that ring is scaled.
* Run 2-3 passes: scaling one ring slightly perturbs neighbours via the
  falloff, so iterating re-converges all targets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RingTarget:
    """One circumference to hit, by mesh Y level."""
    code: str            # seamly code, for logging
    y_level: float       # world-frame Y of the slice (metres)
    target_cm: float     # desired circumference, centimetres
    band_m: float = 0.06 # half-height of the deformed Y-band (metres)
    region_mask: np.ndarray | None = None  # verts allowed to move


def apply_scale_profile(
    verts: np.ndarray,
    y_levels: np.ndarray,
    scales: np.ndarray,
    region_mask: np.ndarray | None = None,
    ramp_m: float = 0.10,
) -> np.ndarray:
    """Apply a *continuous* radial scale profile, killing the surface
    banding that discrete per-ring bumps produce.

    ``y_levels`` / ``scales`` are the control points (one per girth
    target). A monotone-in-Y piecewise-linear curve interpolates the
    scale for every vertex's Y. Outside the control range the scale
    ramps smoothly back to 1.0 over ``ramp_m`` so the deformed torso
    blends into undeformed neighbours (neck, thighs).

    Radial scaling is about a single torso centreline (per-vertex Y has
    its own XZ centre, linearly interpolated between control slices) so
    the cross-section stays centred as girth changes."""
    verts = verts.copy()
    order = np.argsort(y_levels)
    ys = np.asarray(y_levels, dtype=np.float64)[order]
    ss = np.asarray(scales, dtype=np.float64)[order]

    vy = verts[:, 1]
    # Piecewise-linear scale, with smooth ramp-to-1 outside the range.
    prof = np.interp(vy, ys, ss, left=ss[0], right=ss[-1])
    below = vy < ys[0]
    above = vy > ys[-1]
    # Cosine ramp the out-of-range scale back to 1.0.
    d_below = np.clip((ys[0] - vy[below]) / ramp_m, 0, 1)
    prof[below] = 1.0 + (ss[0] - 1.0) * 0.5 * (1 + np.cos(np.pi * d_below))
    d_above = np.clip((vy[above] - ys[-1]) / ramp_m, 0, 1)
    prof[above] = 1.0 + (ss[-1] - 1.0) * 0.5 * (1 + np.cos(np.pi * d_above))

    if region_mask is not None:
        prof = np.where(region_mask, prof, 1.0)

    # Per-Y XZ centroid of the torso, interpolated between control
    # slices, so the radial origin tracks the body centreline.
    if region_mask is not None:
        torso = verts[region_mask]
    else:
        torso = verts
    cx = np.interp(vy, ys,
                   [torso[np.abs(torso[:, 1] - y) < 0.03][:, 0].mean()
                    if np.any(np.abs(torso[:, 1] - y) < 0.03) else 0.0
                    for y in ys])
    cz = np.interp(vy, ys,
                   [torso[np.abs(torso[:, 1] - y) < 0.03][:, 2].mean()
                    if np.any(np.abs(torso[:, 1] - y) < 0.03) else 0.0
                    for y in ys])

    verts[:, 0] = cx + (verts[:, 0] - cx) * prof
    verts[:, 2] = cz + (verts[:, 2] - cz) * prof
    return verts


def apply_radial_scale(
    verts: np.ndarray,
    y_level: float,
    band_m: float,
    scale: float,
    region_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Pure geometric op: scale vertices radially (XZ) about the Y-band's
    centroid by ``scale``, cosine-falloff-weighted over ``band_m``.

    No measurement here — the caller supplies ``scale`` (typically
    target_cm / extractor_current_cm) so the real measurement extractor
    stays the single source of truth and there's no proxy mismatch."""
    verts = verts.copy()
    in_band = np.abs(verts[:, 1] - y_level) < band_m
    sel = in_band & region_mask if region_mask is not None else in_band
    if sel.sum() < 8:
        return verts
    centroid = verts[sel][:, [0, 2]].mean(axis=0)

    dy = np.abs(verts[:, 1] - y_level)
    weight = _cosine_falloff(dy, band_m)
    if region_mask is not None:
        weight = weight * region_mask.astype(weight.dtype)
    per_vertex_scale = 1.0 + weight * (scale - 1.0)

    radial = verts[:, [0, 2]] - centroid[None, :]
    verts[:, 0] = centroid[0] + radial[:, 0] * per_vertex_scale
    verts[:, 2] = centroid[1] + radial[:, 1] * per_vertex_scale
    return verts


def _slice_perimeter_cm(
    verts: np.ndarray,
    y_level: float,
    band: float = 0.012,
    region_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Convex-hull perimeter (cm) of the vertices in a thin Y-band, plus
    the band's XZ centroid. Mirrors PlanarGirth's hull-perimeter measure.

    ``region_mask`` (bool, per-vertex) restricts the slice to a body
    region — pass the SMPL-X torso mask so arms crossing the bust/waist/
    hip Y-bands don't inflate the hull (PlanarGirth uses the same torso
    restriction)."""
    from scipy.spatial import ConvexHull

    m = np.abs(verts[:, 1] - y_level) < band
    if region_mask is not None:
        m = m & region_mask
    pts = verts[m][:, [0, 2]]
    centroid = pts.mean(axis=0) if len(pts) else np.zeros(2)
    if len(pts) < 8:
        return float("nan"), centroid
    try:
        hull = ConvexHull(pts)
    except Exception:  # noqa: BLE001 — degenerate slice
        return float("nan"), centroid
    # scipy ConvexHull.area is the perimeter for 2-D input.
    return float(hull.area) * 100.0, centroid


def _cosine_falloff(dist: np.ndarray, band: float) -> np.ndarray:
    """1.0 at dist=0, smoothly → 0.0 at dist=band, 0 beyond.

    Cosine ramp keeps C1 continuity so the deformed surface has no
    visible crease at the band edge."""
    w = np.zeros_like(dist)
    inside = dist < band
    w[inside] = 0.5 * (1.0 + np.cos(np.pi * dist[inside] / band))
    return w


def deform_ring(
    verts: np.ndarray,
    target: RingTarget,
    *,
    measure_band: float = 0.012,
    region_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """Apply one ring deformation. Returns (new_verts, before_cm, after_cm).

    Vertices within ``target.band_m`` of ``target.y_level`` AND inside
    ``region_mask`` (e.g. torso) are scaled radially (in XZ) about the
    slice centroid by ``target_cm / current``, weighted by a cosine
    falloff so the deformation tapers to zero at the band edge.

    Restricting the deformed set to the torso mask keeps arm vertices
    fixed — scaling them radially from the torso centroid would warp
    the arms."""
    verts = verts.copy()
    current_cm, centroid = _slice_perimeter_cm(
        verts, target.y_level, band=measure_band, region_mask=region_mask)
    if not np.isfinite(current_cm) or current_cm <= 0:
        return verts, current_cm, current_cm

    scale = target.target_cm / current_cm

    dy = np.abs(verts[:, 1] - target.y_level)
    weight = _cosine_falloff(dy, target.band_m)  # (N,)
    if region_mask is not None:
        weight = weight * region_mask.astype(weight.dtype)

    # Per-vertex radial scale: lerp 1.0 → scale by the falloff weight.
    per_vertex_scale = 1.0 + weight * (scale - 1.0)

    # Radial vector in XZ from the slice centroid.
    radial = verts[:, [0, 2]] - centroid[None, :]
    verts[:, 0] = centroid[0] + radial[:, 0] * per_vertex_scale
    verts[:, 2] = centroid[1] + radial[:, 1] * per_vertex_scale

    after_cm, _ = _slice_perimeter_cm(
        verts, target.y_level, band=measure_band, region_mask=region_mask)
    return verts, current_cm, after_cm


def deform_rings(
    verts: np.ndarray,
    targets: list[RingTarget],
    *,
    passes: int = 3,
    tol_cm: float = 0.3,
    region_mask: np.ndarray | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Apply all ring targets, iterating ``passes`` times so the cosine
    falloffs of neighbouring rings re-converge.

    Targets are processed top-to-bottom (descending Y) each pass —
    deterministic order so overlapping bands compose the same way.

    ``region_mask`` restricts both measurement and deformation to a
    body region (pass the SMPL-X torso mask)."""
    verts = verts.copy()
    ordered = sorted(targets, key=lambda t: -t.y_level)

    for p in range(passes):
        max_resid = 0.0
        for t in ordered:
            verts, before, after = deform_ring(
                verts, t, region_mask=region_mask)
            if np.isfinite(after):
                max_resid = max(max_resid, abs(after - t.target_cm))
            if verbose:
                print(f"  pass {p+1} {t.code}: "
                      f"{before:.2f} → {after:.2f} cm "
                      f"(target {t.target_cm:.2f})")
        if max_resid <= tol_cm:
            if verbose:
                print(f"ring deform converged pass {p+1} "
                      f"(max residual {max_resid:.2f} cm)")
            break
    return verts
