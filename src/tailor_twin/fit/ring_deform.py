"""Ring deformation — hit tape-measured circumferences exactly.

Background
----------
A parametric body fit (SMPL-X betas, or any learned shape space) cannot
satisfy an arbitrary set of girth targets simultaneously: the shape
space couples correlated dimensions (e.g. highbust and bust are driven
by the same betas, so you cannot freely set highbust 4 cm below bust).
The chamfer fit gets the body 95 % right; the last few centimetres on
each girth need a *geometric* edit, not another beta solve.

Ring deformation does exactly that. ``ring_deform_cli`` drives it in an
extractor-feedback loop: measure each target girth with the REAL Seamly
extractor, compute a radial scale = target / current per ring, apply one
continuous scale profile over the torso (``apply_scale_profile``; per-leg
girths use ``apply_radial_scale``), repeat until every residual is
< 0.3 cm.

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


def _cosine_falloff(dist: np.ndarray, band: float) -> np.ndarray:
    """1.0 at dist=0, smoothly → 0.0 at dist=band, 0 beyond.

    Cosine ramp keeps C1 continuity so the deformed surface has no
    visible crease at the band edge."""
    w = np.zeros_like(dist)
    inside = dist < band
    w[inside] = 0.5 * (1.0 + np.cos(np.pi * dist[inside] / band))
    return w


def audit_girth_drift(
    before: dict[str, float],
    after: dict[str, float],
    anchored: set[str] | frozenset[str] | list[str] | tuple[str, ...],
    tol_cm: float = 1.0,
) -> dict:
    """Compare full measurement catalogs before/after tape-anchor deformation.

    The ring deform is *supposed* to move only the anchored girths (plus a
    small feathered neighbourhood); this audit makes that verifiable per
    run. It reports every code that changed, marks which were anchored,
    and flags UNANCHORED codes whose |delta| exceeds ``tol_cm`` — those
    are the ones where the deformation pulled the mesh away from the scan
    by more than the tolerance, e.g. a neighbour ring inside the falloff
    band or an implausible tape value dragging the profile.

    Pure dict-in / dict-out so it is unit-testable without the extractor.
    Codes present in only one catalog are ignored (extraction skips are
    not drift). Non-finite values are ignored.

    Returns::

        {
          "tol_cm": 1.0,
          "drift": {code: {"before_cm", "after_cm", "delta_cm",
                            "anchored"}},         # |delta| >= 0.05 only
          "flagged_unanchored": [codes],           # |delta| > tol, sorted
        }
    """
    anchored_set = set(anchored)
    drift: dict[str, dict] = {}
    flagged: list[tuple[float, str]] = []
    for code in sorted(set(before) & set(after)):
        b = float(before[code])
        a = float(after[code])
        if not (np.isfinite(b) and np.isfinite(a)):
            continue
        d = a - b
        is_anchored = code in anchored_set
        if abs(d) >= 0.05 or is_anchored:
            drift[code] = {
                "before_cm": round(b, 2),
                "after_cm": round(a, 2),
                "delta_cm": round(d, 2),
                "anchored": is_anchored,
            }
        if not is_anchored and abs(d) > tol_cm:
            flagged.append((abs(d), code))
    flagged.sort(reverse=True)
    return {
        "tol_cm": tol_cm,
        "drift": drift,
        "flagged_unanchored": [code for _, code in flagged],
    }
