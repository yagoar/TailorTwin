"""Post-fit beta refinement against user-supplied tape measurements.

The chamfer-driven ``fit_scan`` matches mesh surface to scan geometry,
but tight clothing, pose drift, and ARKit pose noise can leave specific
anthropometric circumferences (waist, bust, hip) several centimetres off
the user's real measurements. This module nudges the SMPL-X betas so the
target measurements match the user-supplied tape values, while staying
close to the chamfer-converged betas (anchor regularizer).

Approach: damped Gauss-Newton with finite-difference Jacobian against
the *authoritative* ``extract_catalog`` (same code path measurement CLI
uses), so the refined npz reports values consistent with the rest of
the pipeline.

The body pose, global_orient, transl, and displacement are kept fixed —
we refine shape only. Only the first ``num_betas_active`` betas (default
10, sorted by PCA variance in CAESAR) are optimized; the remaining
high-order betas stay at their chamfer-fit values.

Stopping condition: every target within ``tol_cm`` (default 0.5 cm) or
``max_iters`` reached.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import smplx
import torch

from ..measure.seamly_extractor import extract_catalog


@dataclass
class RefineResult:
    betas: np.ndarray              # refined betas
    smplx_vertices: np.ndarray     # mesh in fit-pose with refined betas
    values_before: dict[str, float]
    values_after: dict[str, float]
    targets: dict[str, float]
    n_iters: int
    converged: bool


def _build_model(model_folder: str, gender: str, num_betas: int) -> smplx.SMPLX:
    return smplx.create(
        model_path=model_folder, model_type="smplx",
        gender=gender, num_betas=num_betas,
        use_pca=False, flat_hand_mean=True, batch_size=1,
    )


def _forward(
    bm: smplx.SMPLX,
    betas: np.ndarray,
    body_pose: np.ndarray,
    global_orient: np.ndarray,
    transl: np.ndarray,
    displacement: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run SMPL-X with the given parameters; return (verts, joints) in
    world frame (with displacement added to verts when supplied)."""
    with torch.no_grad():
        out = bm(
            betas=torch.from_numpy(betas[None, :].astype(np.float32)),
            body_pose=torch.from_numpy(
                body_pose.reshape(1, -1).astype(np.float32)),
            global_orient=torch.from_numpy(
                global_orient.reshape(1, -1).astype(np.float32)),
            transl=torch.from_numpy(transl.reshape(1, 3).astype(np.float32)),
            return_full_pose=False,
        )
    v = out.vertices[0].cpu().numpy()
    j = out.joints[0].cpu().numpy()
    if displacement is not None and displacement.shape == v.shape:
        v = v + displacement
    return v, j


def _eval_targets(
    verts: np.ndarray,
    faces: np.ndarray,
    target_codes: tuple[str, ...],
    gender: str,
    waist_y_override: float | None,
    joints: np.ndarray | None = None,
    waist_height_cm: float | None = None,
) -> dict[str, float]:
    """Run the full extractor and pluck out the requested codes.

    ``waist_height_cm`` (floor-relative tape value) is resolved against the
    CURRENT verts on every call — the floor Y shifts as betas change leg
    length, so a once-computed absolute Y would drift over the iterations.
    An explicit ``waist_y_override`` wins when both are set.
    """
    if waist_y_override is None and waist_height_cm is not None:
        from ..measure.landmarks import waist_y_from_height
        waist_y_override = waist_y_from_height(verts, waist_height_cm)
    cat = extract_catalog(
        verts, faces, joints=joints,
        waist_y_override=waist_y_override, gender=gender,
    )
    out: dict[str, float] = {}
    for code in target_codes:
        if code in cat.values:
            out[code] = float(cat.values[code])
        else:
            reason = cat.skipped.get(code, "not produced")
            raise KeyError(
                f"target code {code!r} not produced by extractor: {reason}")
    return out


def _build_a_pose(shoulder_deg: float) -> np.ndarray:
    """Return a (21,3) body_pose axis-angle array for A-pose: SMPL-X
    canonical with both shoulders rotated about Z so the arms swing
    down by ``shoulder_deg`` degrees. ``shoulder_deg`` = 0 → T-pose,
    typical A-pose ≈ 30°."""
    pose = np.zeros((21, 3), dtype=np.float64)
    rad = np.deg2rad(shoulder_deg)
    # body_pose indices: 15 = L_Shoulder, 16 = R_Shoulder.
    # SMPL-X coords: +Y up. Rotation about Z swings arm in the XY plane.
    # Empirically: L_Shoulder sign + brings arm down toward -Y;
    # R_Shoulder needs opposite sign. If your viewer shows arms going
    # up, flip the sign of shoulder_deg or pass a negative value.
    pose[15, 2] = -rad   # L_Shoulder
    pose[16, 2] = +rad   # R_Shoulder
    return pose


def refine_betas_to_tape(
    fit_npz: Path,
    targets: Mapping[str, float],     # cm, keyed by seamly code
    *,
    model_folder: str = "data/body_models",
    num_betas_active: int = 10,
    max_iters: int = 12,
    tol_cm: float = 0.5,
    step_clip: float = 0.4,           # per-iter |delta beta| bound
    anchor_weight: float = 0.05,      # pulls towards original betas
    ridge: float = 0.01,              # JᵀJ damping
    waist_y_override: float | None = None,
    waist_height_cm: float | None = None,
    a_pose_shoulder_deg: float = 0.0,
    verbose: bool = True,
) -> RefineResult:
    """Adjust the first ``num_betas_active`` betas of a fit so the
    extractor reports values matching ``targets`` (cm).

    The pose, translation and displacement from the original npz are
    preserved — only shape changes. The returned ``smplx_vertices`` are
    the deformed mesh in the same pose as the input fit.
    """
    fit = np.load(fit_npz)
    betas0 = fit["betas"].astype(np.float64)
    # Refinement runs in a canonical pose so the optimized betas match
    # the measurements the user tape-measured (pose-agnostic). The fit's
    # original body_pose (e.g. hand-on-hip from the scan) is discarded —
    # the refined fit is saved in this canonical pose for garment use.
    # T-pose (0°) or A-pose (≈30°) selectable via a_pose_shoulder_deg.
    body_pose = _build_a_pose(a_pose_shoulder_deg)
    global_orient = np.zeros((3,), dtype=np.float64)
    transl = np.zeros((3,), dtype=np.float64)
    disp = fit["displacement"] if "displacement" in fit.files else None

    from .fit import fit_gender
    gender = fit_gender(fit)

    num_betas = betas0.shape[0]
    n_active = min(num_betas_active, num_betas)
    target_codes = tuple(targets.keys())
    target_vec = np.array([targets[c] for c in target_codes], dtype=np.float64)

    bm = _build_model(model_folder, gender, num_betas)
    faces = np.asarray(bm.faces, dtype=np.int32)

    betas = betas0.copy()
    v0, j0 = _forward(bm, betas, body_pose, global_orient, transl, disp)
    values_before = _eval_targets(
        v0, faces, target_codes, gender, waist_y_override, joints=j0,
        waist_height_cm=waist_height_cm)
    if verbose:
        print(f"refine: gender={gender}, active betas={n_active}, "
              f"targets={dict(zip(target_codes, target_vec))}")
        print(f"  iter 00  before: {values_before}")

    eps = 0.05
    converged = False
    iter_count = 0
    for it in range(1, max_iters + 1):
        iter_count = it
        verts, jts = _forward(
            bm, betas, body_pose, global_orient, transl, disp)
        current = _eval_targets(
            verts, faces, target_codes, gender, waist_y_override, joints=jts,
            waist_height_cm=waist_height_cm)
        residual = target_vec - np.array(
            [current[c] for c in target_codes], dtype=np.float64)
        max_abs = float(np.max(np.abs(residual)))
        if verbose:
            row = "  ".join(f"{c}={current[c]:.2f}(Δ{residual[i]:+.2f})"
                            for i, c in enumerate(target_codes))
            print(f"  iter {it:02d}  {row}")
        if max_abs <= tol_cm:
            converged = True
            break

        # Finite-difference Jacobian on active betas.
        J = np.zeros((len(target_codes), n_active), dtype=np.float64)
        for j in range(n_active):
            betas_p = betas.copy()
            betas_p[j] += eps
            v_p, j_p = _forward(
                bm, betas_p, body_pose, global_orient, transl, disp)
            try:
                m_p = _eval_targets(
                    v_p, faces, target_codes, gender, waist_y_override,
                    joints=j_p, waist_height_cm=waist_height_cm)
            except KeyError as e:
                if verbose:
                    print(f"    perturb beta[{j}] skipped: {e}")
                continue
            for i, c in enumerate(target_codes):
                J[i, j] = (m_p[c] - current[c]) / eps

        # Damped Gauss-Newton + anchor.
        # Build augmented system:
        #   [ J ]              [ residual ]
        #   [ √α·I ] · δβ_a = [    0    ]
        # where I is on the active block, weighted by anchor_weight.
        n = n_active
        sqrt_alpha = np.sqrt(anchor_weight)
        A = np.vstack([J, sqrt_alpha * np.eye(n)])
        # Bias the anchor toward the *original* betas, not the current ones,
        # so the regularizer prevents long-term drift, not motion itself.
        anchor_offset = betas0[:n] - betas[:n]
        rhs = np.concatenate([residual, sqrt_alpha * anchor_offset])
        # Solve with ridge.
        AtA = A.T @ A + ridge * np.eye(n)
        Atb = A.T @ rhs
        try:
            delta = np.linalg.solve(AtA, Atb)
        except np.linalg.LinAlgError as e:
            print(f"  Gauss-Newton solve failed ({e}); stopping")
            break

        # Step-size clip (per-component bound).
        delta = np.clip(delta, -step_clip, step_clip)
        betas[:n] += delta

    # Final values + verts.
    final_verts, final_joints = _forward(
        bm, betas, body_pose, global_orient, transl, disp)
    final_vals = _eval_targets(
        final_verts, faces, target_codes, gender, waist_y_override,
        joints=final_joints, waist_height_cm=waist_height_cm)

    if verbose:
        if converged:
            print(f"refine: converged in {iter_count} iters within {tol_cm} cm")
        else:
            print(f"refine: stopped at {iter_count} iters (not converged)")

    return RefineResult(
        betas=betas.astype(np.float32),
        smplx_vertices=final_verts.astype(np.float32),
        values_before=values_before,
        values_after=final_vals,
        targets=dict(zip(target_codes, target_vec)),
        n_iters=iter_count,
        converged=converged,
    )


def save_refined_fit(
    src_fit_npz: Path,
    refined: RefineResult,
    out_npz: Path,
    *,
    canonical_pose: bool = True,
    a_pose_shoulder_deg: float = 0.0,
    model_folder: str = "data/body_models",
) -> None:
    """Write a new fit npz that mirrors the source but with refined betas.

    ``canonical_pose=True`` (default) zeroes body_pose, global_orient
    and transl, then recomputes vertices + joints. This produces a
    clean T-pose body — the standard input for garment patterns and
    visual inspection. The bent-arm re-pose step still works because
    it offsets from the stored body_pose (zero here)."""
    src = np.load(src_fit_npz)
    payload = {k: src[k] for k in src.files}
    payload["betas"] = refined.betas

    if canonical_pose:
        from .fit import fit_gender
        gender = fit_gender(src)
        num_betas = refined.betas.shape[0]
        bm = _build_model(model_folder, gender, num_betas)
        canon_pose = _build_a_pose(a_pose_shoulder_deg).astype(np.float32)
        zero_orient = np.zeros((3,), dtype=np.float32)
        zero_transl = np.zeros((3,), dtype=np.float32)
        disp = src["displacement"] if "displacement" in src.files else None
        verts_t, joints_t = _forward(
            bm, refined.betas, canon_pose, zero_orient, zero_transl, disp)
        payload["smplx_vertices"] = verts_t.astype(np.float32)
        payload["smplx_joints"] = joints_t.astype(np.float32)
        payload["body_pose"] = canon_pose
        payload["global_orient"] = zero_orient
        payload["transl"] = zero_transl
        # z (VPoser latent) becomes meaningless once body_pose is zeroed;
        # store empty array so legacy readers still parse the npz.
        payload["z"] = np.array([])
    else:
        payload["smplx_vertices"] = refined.smplx_vertices

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **payload)
