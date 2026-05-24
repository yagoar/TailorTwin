"""Fit SMPL-X shape betas to two silhouette photos.

This is the parametric core of ``silhouette-fit``. Given the front and
side silhouette profiles (per-Y body width / depth from the photos), it
solves for the SMPL-X shape parameters whose A-pose mesh reproduces
those outlines.

Why parametric, not free-form
-----------------------------
An earlier version scaled mesh vertices per horizontal slice to hit the
silhouette extents exactly. That hits the *numbers* but pushes vertices
off the body manifold — the result has dents, over-pinched waists and a
seam where the deformed torso meets undeformed limbs.

Estimating betas instead keeps the body inside the CAESAR-trained SMPL-X
shape space, so every iterate is an anatomically plausible body. This is
the approach the literature converges on — e.g. Škorvánková et al.
(arXiv:2205.14347) explicitly fit SMPL betas rather than free-form
deform, "constraining outputs to valid shape space". The cost is that an
arbitrary silhouette may not be reachable exactly; a residual of 1-2 cm
on some girths is expected and is left for an optional ring-deform
polish.

Solver
------
Damped Gauss-Newton with a finite-difference Jacobian, mirroring
``refine_to_tape``. The residual is dense — body width and depth at
``n_slices`` torso heights (≈48 residuals) against ``n_active`` betas
(≈20) — so the system is well over-determined and the fit captures the
*shape*, not just a few girths. Heights are compared in normalized Y
(0 = feet, 1 = crown) so the fit is invariant to overall stature; the
caller applies the absolute height as a final uniform Y-scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import smplx
import torch

from .refine_to_tape import _build_a_pose


@dataclass
class BetaFitResult:
    betas: np.ndarray              # fitted full beta vector
    width_before: np.ndarray       # per-slice cm, before
    width_after: np.ndarray
    depth_before: np.ndarray
    depth_after: np.ndarray
    width_target: np.ndarray
    depth_target: np.ndarray
    y_norm: np.ndarray             # torso slice normalized heights
    n_iters: int
    converged: bool
    # Leg slices (None when no leg profiles were supplied).
    leg_y_norm: np.ndarray | None = None
    leg_width_after: np.ndarray | None = None
    leg_depth_after: np.ndarray | None = None
    leg_width_target: np.ndarray | None = None
    leg_depth_target: np.ndarray | None = None
    # Arm width profile, shoulder→wrist (None when no arm profile given).
    arm_width_after: np.ndarray | None = None
    arm_width_target: np.ndarray | None = None


def _forward_verts(
    bm: smplx.SMPLX, betas: np.ndarray, body_pose: np.ndarray,
) -> np.ndarray:
    """SMPL-X A-pose vertices for ``betas`` — pure parametric, no
    per-vertex displacement (a plausible body is the whole point)."""
    with torch.no_grad():
        out = bm(
            betas=torch.from_numpy(betas[None, :].astype(np.float32)),
            body_pose=torch.from_numpy(
                body_pose.reshape(1, -1).astype(np.float32)),
            global_orient=torch.zeros(1, 3),
            transl=torch.zeros(1, 3),
            return_full_pose=False,
        )
    return out.vertices[0].cpu().numpy().astype(np.float64)


def _arm_extents(
    verts: np.ndarray, arm_mask: np.ndarray, n_bins: int,
) -> np.ndarray:
    """Mesh arm width profile (cm, shoulder→wrist), measured the same
    way as the silhouette arm: front projection (X, Y) → PCA axis →
    perpendicular slice widths."""
    from .silhouette import _axis_profile
    pts = verts[arm_mask][:, [0, 1]]
    return _axis_profile(pts, n_bins, y_up=True) * 100.0


def ellipse_perim(width: float, depth: float) -> float:
    """Ramanujan ellipse perimeter for bounding-box ``width`` × ``depth``
    — the girth a width/depth pair implies for an elliptical section."""
    a, b = width / 2.0, depth / 2.0
    return float(np.pi * (3 * (a + b)
                          - np.sqrt((3 * a + b) * (a + 3 * b))))


def _hull_perim(pts2d: np.ndarray) -> float:
    """Convex-hull perimeter of a set of 2-D points — the same girth
    estimate the measurement extractor's PlanarGirth uses on a ring."""
    if len(pts2d) < 3:
        return np.nan
    from scipy.spatial import ConvexHull
    try:
        h = ConvexHull(pts2d)
    except Exception:
        return np.nan
    v = pts2d[h.vertices]
    d = np.diff(np.vstack([v, v[:1]]), axis=0)
    return float(np.sqrt((d ** 2).sum(axis=1)).sum())


def _slice_extents(
    verts: np.ndarray,
    torso_mask: np.ndarray,
    y_norm: np.ndarray,
    band_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-slice torso width (X), depth (Z) and girth, all in cm.

    Heights are normalized: slice ``g`` sits at ``y_min + g·span``. The
    extent is the 1st-99th percentile spread of torso verts in a Y-band,
    robust to a few stray vertices. Girth is the convex-hull perimeter
    of the band's (X, Z) points — width/depth pin the bounding box,
    girth pins the actual ring length so the mesh section cannot round
    out fuller than the body."""
    y = verts[:, 1]
    y_min, y_max = float(y.min()), float(y.max())
    span = y_max - y_min
    band = max(band_frac * span, 0.012)
    width = np.full(len(y_norm), np.nan)
    depth = np.full(len(y_norm), np.nan)
    girth = np.full(len(y_norm), np.nan)
    for i, g in enumerate(y_norm):
        yw = y_min + g * span
        sel = (np.abs(y - yw) < band) & torso_mask
        if sel.sum() < 8:
            continue
        xs = verts[sel][:, 0]
        zs = verts[sel][:, 2]
        width[i] = (np.percentile(xs, 99) - np.percentile(xs, 1)) * 100.0
        depth[i] = (np.percentile(zs, 99) - np.percentile(zs, 1)) * 100.0
        girth[i] = _hull_perim(verts[sel][:, [0, 2]] * 100.0)
    return width, depth, girth


def fit_betas_to_silhouette(
    base_betas: np.ndarray,
    gender: str,
    front_prof,
    side_prof,
    *,
    height_cm: float,
    front_leg_prof=None,
    side_leg_prof=None,
    arm_prof=None,
    n_arm_bins: int = 10,
    model_folder: str = "data/body_models",
    a_pose_shoulder_deg: float = 30.0,
    n_active: int = 20,
    n_slices: int = 24,
    y_lo: float = 0.50,
    y_hi: float = 0.82,
    leg_lo: float = 0.12,
    leg_hi: float = 0.42,   # stop below the crotch — thighs merge above
    n_leg_slices: int = 14,
    max_iters: int = 18,
    tol_cm: float = 0.25,
    anchor_weight: float = 0.02,
    ridge: float = 0.02,
    step_clip: float = 0.6,
    girth_w: float = 0.0,
    verbose: bool = True,
) -> BetaFitResult:
    """Solve SMPL-X betas so the A-pose mesh matches the silhouettes.

    ``front_prof`` / ``side_prof`` are
    :class:`~tailor_twin.fit.silhouette.SilhouetteProfile` objects. Only
    the first ``n_active`` betas move (the dominant CAESAR PCA modes);
    higher-order betas stay at ``base_betas``. An anchor regularizer
    keeps the active betas near the base so the solve does not chase
    silhouette noise into an extreme body.
    """
    from ..fit.silhouette import sample_extent_cm, sample_valid

    num_betas = base_betas.shape[0]
    n_active = min(n_active, num_betas)
    bm = smplx.create(
        model_path=model_folder, model_type="smplx", gender=gender,
        num_betas=num_betas, use_pca=False, flat_hand_mean=True,
        batch_size=1)
    body_pose = _build_a_pose(a_pose_shoulder_deg)

    from ..measure.regions import region_vertex_mask
    torso_mask = region_vertex_mask(("torso",), model_folder=model_folder,
                                    gender=gender)

    y_norm = np.linspace(y_lo, y_hi, n_slices)
    band_frac = 0.5 * (y_hi - y_lo) / max(n_slices - 1, 1)

    # Targets (cm) from the silhouette profiles, sampled at slice heights.
    width_tgt = np.array([sample_extent_cm(front_prof, g) for g in y_norm])
    depth_tgt = np.array([sample_extent_cm(side_prof, g) for g in y_norm])
    width_ok = np.array([sample_valid(front_prof, g) for g in y_norm])
    depth_ok = np.array([sample_valid(side_prof, g) for g in y_norm])

    # Girth target — the ellipse perimeter implied by each slice's
    # width/depth pair. Width and depth alone pin only the bounding box;
    # the SMPL-X section can still bow out fuller (the bust/highbust
    # over-read). When ``girth_w > 0`` a perimeter residual is added so
    # the mesh ring length is held to the photo-measured girth too.
    girth_tgt = np.array([ellipse_perim(wt, dt)
                          for wt, dt in zip(width_tgt, depth_tgt)])

    # Leg slices — photo-driven thighs/calves. The left_leg SMPL-X
    # region is measured; the silhouette leg profiles each read a single
    # leg blob (extract_profile pick="widest"). Both legs share the same
    # betas, so fitting one leg shapes both.
    legs_on = front_leg_prof is not None and side_leg_prof is not None
    leg_mask = leg_y = leg_band = leg_w_tgt = leg_d_tgt = None
    if legs_on:
        leg_mask = region_vertex_mask(("left_leg",),
                                      model_folder=model_folder, gender=gender)
        leg_y = np.linspace(leg_lo, leg_hi, n_leg_slices)
        leg_band = 0.5 * (leg_hi - leg_lo) / max(n_leg_slices - 1, 1)
        leg_w_tgt = np.array([sample_extent_cm(front_leg_prof, g)
                              for g in leg_y])
        leg_d_tgt = np.array([sample_extent_cm(side_leg_prof, g)
                              for g in leg_y])
        # Front leg width is only valid where the two legs are separated
        # (see extract_profile); touching-thigh slices keep the base
        # shape for width but are still depth-constrained from the side.
        leg_w_ok = np.array([sample_valid(front_leg_prof, g)
                             for g in leg_y])

    # Arm width profile — left_arm region matched to the silhouette arm
    # profile (shoulder→wrist). Both arms share betas, so one suffices.
    arms_on = arm_prof is not None
    arm_mask = None
    if arms_on:
        arm_mask = region_vertex_mask(("left_arm",),
                                      model_folder=model_folder, gender=gender)
        arm_prof = np.asarray(arm_prof, dtype=np.float64)

    # The silhouette extents are X/Z only — they do not pin overall
    # stature, so the betas could shrink the whole body and let a final
    # Y-scale "fix" the height (which would warp proportions). A height
    # residual keeps absolute scale honest inside the solve.
    height_w = 2.0

    def residual(betas: np.ndarray):
        verts = _forward_verts(bm, betas, body_pose)
        w, d, gth = _slice_extents(verts, torso_mask, y_norm, band_frac)
        # Drop slices the mesh or the photo could not measure.
        wmask = width_ok & np.isfinite(w)
        dmask = depth_ok & np.isfinite(d)
        mesh_h = (verts[:, 1].max() - verts[:, 1].min()) * 100.0
        parts = [width_tgt[wmask] - w[wmask],
                 depth_tgt[dmask] - d[dmask],
                 np.array([height_w * (height_cm - mesh_h)])]
        if girth_w > 0.0:
            gm = (width_ok & depth_ok & np.isfinite(gth)
                  & np.isfinite(girth_tgt))
            parts.append(girth_w * (girth_tgt[gm] - gth[gm]))
        lw = ld = aw = None
        if legs_on:
            lw, ld, _ = _slice_extents(verts, leg_mask, leg_y, leg_band)
            lwm = leg_w_ok & np.isfinite(lw)
            ldm = np.isfinite(ld)
            parts.append(leg_w_tgt[lwm] - lw[lwm])
            parts.append(leg_d_tgt[ldm] - ld[ldm])
        if arms_on:
            aw = _arm_extents(verts, arm_mask, n_arm_bins)
            am = np.isfinite(aw) & np.isfinite(arm_prof)
            parts.append(arm_prof[am] - aw[am])
        return np.concatenate(parts), w, d, lw, ld, aw

    betas = base_betas.astype(np.float64).copy()
    res0, w0, d0, _, _, _ = residual(betas)
    if verbose:
        print(f"betas fit: gender={gender}, active={n_active}, "
              f"slices={n_slices}, residuals={res0.size}")
        print(f"  iter 00  RMS {np.sqrt(np.mean(res0**2)):.2f} cm  "
              f"max {np.max(np.abs(res0)):.2f} cm")

    # Levenberg-Marquardt. A plain damped Gauss-Newton step is applied
    # unconditionally, so when the silhouette is not exactly reachable in
    # shape space the iterate oscillates — RMS stays flat near the
    # minimum while the betas (and the girths) wander, and the result
    # then depends on ``max_iters``. LM only accepts a step that lowers
    # the augmented cost, growing the damping ``lam`` until it does, so
    # the cost decreases monotonically and the fit settles.
    eps = 0.04
    sa = np.sqrt(anchor_weight)

    def aug_cost(b, r):
        """Data SSE + anchor SSE — the quantity LM must decrease."""
        anc = base_betas[:n_active] - b[:n_active]
        return float(r @ r + anchor_weight * (anc @ anc))

    lam = max(ridge, 1e-3)
    converged = False
    res, w, d, lw, ld, aw = residual(betas)
    cur_cost = aug_cost(betas, res)
    best_betas, best_cost = betas.copy(), cur_cost
    it = 0
    for it in range(1, max_iters + 1):
        max_abs = float(np.max(np.abs(res))) if res.size else 0.0
        rms = float(np.sqrt(np.mean(res**2))) if res.size else 0.0
        if verbose:
            print(f"  iter {it:02d}  RMS {rms:.2f} cm  max {max_abs:.2f} cm  "
                  f"lam {lam:.3g}")
        if max_abs <= tol_cm:
            converged = True
            break

        # Finite-difference Jacobian on the active betas.
        J = np.zeros((res.size, n_active))
        for j in range(n_active):
            bp = betas.copy()
            bp[j] += eps
            rj, *_ = residual(bp)
            if rj.size == res.size:
                J[:, j] = (res - rj) / eps   # d res / d beta
        anchor = base_betas[:n_active] - betas[:n_active]
        rhs = J.T @ res + anchor_weight * anchor
        JtJ = J.T @ J + anchor_weight * np.eye(n_active)

        # Grow lam until the step lowers the augmented cost.
        stepped = False
        for _ in range(8):
            try:
                delta = np.linalg.solve(JtJ + lam * np.eye(n_active), rhs)
            except np.linalg.LinAlgError:
                lam *= 4.0
                continue
            delta = np.clip(delta, -step_clip, step_clip)
            cand = betas.copy()
            cand[:n_active] += delta
            cres, *cextra = residual(cand)
            ccost = aug_cost(cand, cres)
            if ccost < cur_cost:
                betas, res = cand, cres
                w, d, lw, ld, aw = cextra
                cur_cost = ccost
                lam = max(lam * 0.5, 1e-4)
                stepped = True
                if ccost < best_cost:
                    best_cost, best_betas = ccost, betas.copy()
                break
            lam *= 4.0
        if not stepped:
            if verbose:
                print("  no cost-reducing step found; converged")
            converged = True
            break

    betas = best_betas
    res_f, w_f, d_f, lw_f, ld_f, aw_f = residual(betas)
    if verbose:
        tag = "converged" if converged else "stopped"
        print(f"betas fit {tag} after {it} iters "
              f"(max residual {np.max(np.abs(res_f)):.2f} cm)")

    return BetaFitResult(
        betas=betas, width_before=w0, width_after=w_f,
        depth_before=d0, depth_after=d_f,
        width_target=width_tgt, depth_target=depth_tgt,
        y_norm=y_norm, n_iters=it, converged=converged,
        leg_y_norm=leg_y, leg_width_after=lw_f, leg_depth_after=ld_f,
        leg_width_target=leg_w_tgt, leg_depth_target=leg_d_tgt,
        arm_width_after=aw_f, arm_width_target=arm_prof if arms_on else None)
