"""Post-fit cleanup of the SMPL-X+D body.

A raw chamfer+displacement fit faithfully reproduces the scan — including
its defects: left/right asymmetry from uneven orbit coverage, warped face
geometry where LiDAR couldn't see hair, and noisy fingers the 192x256
depth can't resolve. None of those regions carry a drafting measurement,
so we clean them before measurement/export:

  1. Symmetrize the displacement field across the sagittal plane. SMPL-X
     is bilaterally symmetric (and pattern blocks are drafted symmetric),
     so mirroring + averaging removes scan-noise asymmetry while keeping
     real shape. Uses the exact left/right vertex correspondence derived
     from the symmetric template.
  2. Zero the displacement on the head and hands (sphere masks around the
     head and wrist joints). These revert to the clean SMPL-X template —
     measurement-safe: no code measures face/finger geometry, the neck
     girth region sits below the head sphere, and total height is anchored
     separately. Removes the warped-face / splayed-finger artifacts.
  3. Re-pose to the canonical A-pose (30 deg arms) so the exported body is
     pose-normalized and consistent regardless of the scan-time pose.

The result overwrites the fit npz's ``smplx_vertices`` / ``smplx_joints``
/ ``displacement`` / ``body_pose`` so the downstream measure + tape-anchor
stages all read the cleaned, canonical body.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# SMPL-X body-joint indices.
_HEAD_JOINT = 15
_LEFT_ELBOW = 18
_RIGHT_ELBOW = 19
_LEFT_WRIST = 20
_RIGHT_WRIST = 21

DEFAULT_HEAD_RADIUS_M = 0.12   # sphere around the head joint (keeps the neck)
# Feather half-width: the displacement fades 1->0 over [radius-feather,
# radius+feather] instead of switching abruptly, so there is no crease at
# the boundary (a hard step in the displacement field shows as a seam).
DEFAULT_HEAD_FEATHER_M = 0.04
# Hand cut along the elbow->wrist axis: t is the normalised projection
# (0 at elbow, 1 at the wrist joint, >1 into the hand). Displacement is
# kept on the whole forearm up to just before the wrist (t < HAND_T_IN) and
# faded to the clean SMPL-X template hand from the wrist bone down (t >=
# HAND_T_OUT), with only a small feather straddling the wrist so there is no
# seam. The hand is unresolvable at 192x256 and its scan displacement is
# noise; the forearm keeps its real shape. Lateral guard keeps the falloff
# from touching the nearby hip in the A-pose.
DEFAULT_FOREARM_T_IN = 0.95
DEFAULT_FOREARM_T_OUT = 1.05
DEFAULT_FOREARM_LATERAL_M = 0.13
DEFAULT_APOSE_DEG = 30.0


def build_symmetry_map(v_template: np.ndarray) -> np.ndarray:
    """Left/right vertex correspondence from a bilaterally-symmetric mesh.

    For each vertex, returns the index of the vertex nearest its X-mirrored
    position. On the SMPL-X template this is exact (sub-0.1 mm).
    """
    from scipy.spatial import cKDTree

    mirrored = v_template.copy()
    mirrored[:, 0] *= -1.0
    _, sym = cKDTree(v_template).query(mirrored, k=1)
    return sym.astype(np.int64)


def symmetrize_displacement(D: np.ndarray, sym: np.ndarray) -> np.ndarray:
    """Average each vertex's displacement with its X-mirrored partner's.

    The partner's X component is flipped before averaging, so the result is
    bilaterally symmetric by construction.
    """
    flip = np.array([-1.0, 1.0, 1.0])
    return 0.5 * (D + flip * D[sym])


def _sphere_falloff(
    verts: np.ndarray, centre: np.ndarray, radius: float, feather: float,
) -> np.ndarray:
    """Smoothstep weight: 0 inside ``radius-feather``, 1 outside
    ``radius+feather``, C1-smooth between. Used to fade displacement to zero
    across a region boundary without a step."""
    d = np.linalg.norm(verts - centre, axis=1)
    t = np.clip((d - (radius - feather)) / (2.0 * feather), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _forearm_keep(
    verts: np.ndarray, elbow: np.ndarray, wrist: np.ndarray,
    *, t_in: float, t_out: float, lateral: float,
) -> np.ndarray:
    """Keep weight for one forearm: 1 proximal (near elbow), 0 over the
    distal forearm + hand, smooth between. Parameterised by the normalised
    projection ``t`` onto the elbow->wrist axis (0 at elbow, 1 at wrist,
    >1 beyond into the hand). A lateral-distance guard restricts the effect
    to vertices near the arm axis so the nearby hip/torso is untouched.
    """
    axis = wrist - elbow
    L2 = float(axis @ axis) or 1e-9
    rel = verts - elbow
    t = (rel @ axis) / L2
    proj = elbow + np.outer(np.clip(t, 0.0, 1.6), axis)
    lat = np.linalg.norm(verts - proj, axis=1)
    s = np.clip((t - t_in) / (t_out - t_in), 0.0, 1.0)
    clean = s * s * (3.0 - 2.0 * s)          # 0 proximal -> 1 distal/hand
    keep = 1.0 - clean
    return np.where(lat < lateral, keep, 1.0)


def displacement_keep_weight(
    verts: np.ndarray,
    joints: np.ndarray,
    *,
    head_radius: float = DEFAULT_HEAD_RADIUS_M,
    head_feather: float = DEFAULT_HEAD_FEATHER_M,
    forearm_t_in: float = DEFAULT_FOREARM_T_IN,
    forearm_t_out: float = DEFAULT_FOREARM_T_OUT,
    forearm_lateral: float = DEFAULT_FOREARM_LATERAL_M,
) -> np.ndarray:
    """Per-vertex displacement-keep weight in [0, 1].

    0 where scan displacement is unreliable and re-posing warps it — the
    head (sphere around the head joint) and the distal forearm + hands
    (along each elbow->wrist axis) — and 1 on the torso/legs/upper-arm where
    the fit is trustworthy and measurements live. Smooth transitions
    throughout (no seam). The weight is the product of the per-region
    falloffs, so any clean region pulls it to 0 while the rest stays at 1.
    """
    w = _sphere_falloff(verts, joints[_HEAD_JOINT], head_radius, head_feather)
    w *= _forearm_keep(verts, joints[_LEFT_ELBOW], joints[_LEFT_WRIST],
                       t_in=forearm_t_in, t_out=forearm_t_out,
                       lateral=forearm_lateral)
    w *= _forearm_keep(verts, joints[_RIGHT_ELBOW], joints[_RIGHT_WRIST],
                       t_in=forearm_t_in, t_out=forearm_t_out,
                       lateral=forearm_lateral)
    return w


def clean_displacement(
    D: np.ndarray,
    verts: np.ndarray,
    joints: np.ndarray,
    sym: np.ndarray,
    **kw,
) -> np.ndarray:
    """Symmetrize the displacement, then feather it to zero on the head +
    distal forearm/hands.

    The feathered weight avoids the seam a hard mask leaves: the
    displacement ramps smoothly to zero across each region, so the surface
    stays continuous into the cleaned template.
    """
    D_sym = symmetrize_displacement(D, sym)
    w = displacement_keep_weight(verts, joints, **kw)
    return D_sym * w[:, None]


def clean_fit_npz(
    fit_npz: Path,
    out_npz: Path,
    *,
    model_folder: str = "data/body_models",
    gender: str | None = None,
    num_betas: int = 300,
    pose_deg: float = DEFAULT_APOSE_DEG,
    head_radius: float = DEFAULT_HEAD_RADIUS_M,
    verbose: bool = True,
) -> Path:
    """Load a fit npz, clean its displacement, re-pose to canonical A-pose,
    and write the result.

    The cleaned body is regenerated from ``betas`` + the canonical A-pose +
    the cleaned displacement, in the SMPL-X canonical frame (global_orient =
    0, transl = 0) so downstream landmarking sees a pose-normalized body.
    """
    import smplx
    import torch

    from .fit import fit_gender
    from .refine_to_tape import _build_a_pose

    fit = np.load(fit_npz)
    g = gender or fit_gender(fit)
    # flat_hand_mean=False so the un-posed hands take MANO's relaxed mean
    # pose (a natural slight curl) instead of the flat, splayed default that
    # reads as "broken fingers". The fit never optimises hand pose and the
    # 192x256 depth can't resolve fingers, so the relaxed template hand is
    # the best available — and it's measurement-safe: the finger curl is
    # distal to the wrist, so wrist girth (L15) and arm length (to the wrist
    # landmark) are unchanged.
    bm = smplx.create(model_path=model_folder, model_type="smplx", gender=g,
                      num_betas=num_betas, use_pca=False, flat_hand_mean=False,
                      batch_size=1)

    sym = build_symmetry_map(
        bm.v_template.detach().cpu().numpy().astype(np.float64))

    betas = fit["betas"].astype(np.float32)
    canon_pose = _build_a_pose(pose_deg).astype(np.float32)
    with torch.no_grad():
        out = bm(
            betas=torch.from_numpy(betas[None, :]),
            body_pose=torch.from_numpy(canon_pose.reshape(1, -1)),
            global_orient=torch.zeros(1, 3),
            transl=torch.zeros(1, 3),
        )
    verts = out.vertices[0].cpu().numpy().astype(np.float64)
    joints = out.joints[0].cpu().numpy().astype(np.float32)

    D = (fit["displacement"].astype(np.float64)
         if "displacement" in fit.files else np.zeros_like(verts))
    if D.shape == verts.shape and np.any(D):
        D_clean = clean_displacement(D, verts, joints, sym,
                                     head_radius=head_radius)
        if verbose:
            asym = np.abs(D - symmetrize_displacement(D, sym)).mean() * 1000
            w = displacement_keep_weight(verts, joints, head_radius=head_radius)
            nz = int((w < 0.5).sum())
            print(f"  clean-fit: symmetrized D (mean asym {asym:.2f}mm), "
                  f"feathered out ~{nz} head/forearm/hand verts, canonical "
                  f"A-pose {pose_deg:.0f}deg")
    else:
        D_clean = np.zeros_like(verts)
        if verbose:
            print(f"  clean-fit: no displacement; canonical A-pose "
                  f"{pose_deg:.0f}deg only")
    verts_clean = verts + D_clean

    payload = {k: fit[k] for k in fit.files}
    payload["smplx_vertices"] = verts_clean.astype(np.float32)
    payload["smplx_joints"] = joints
    payload["displacement"] = D_clean.astype(np.float32)
    payload["body_pose"] = canon_pose
    payload["global_orient"] = np.zeros((3,), dtype=np.float32)
    payload["transl"] = np.zeros((3,), dtype=np.float32)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **payload)
    return out_npz
