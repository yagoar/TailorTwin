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


# SMPL-X body-joint indices used for the region spheres.
_HEAD_JOINT = 15
_LEFT_WRIST = 20
_RIGHT_WRIST = 21

DEFAULT_HEAD_RADIUS_M = 0.12   # sphere around the head joint (keeps the neck)
DEFAULT_HAND_RADIUS_M = 0.10   # sphere around each wrist (wrist-out = hand)
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


def head_hand_mask(
    verts: np.ndarray,
    joints: np.ndarray,
    *,
    head_radius: float = DEFAULT_HEAD_RADIUS_M,
    hand_radius: float = DEFAULT_HAND_RADIUS_M,
) -> np.ndarray:
    """Boolean per-vertex mask of head + both hands (sphere around joints).

    Spheres (not a Y-cut) so the neck and shoulders are preserved: the head
    sphere is centred on the head joint and the hand spheres on the wrists.
    """
    head = np.linalg.norm(verts - joints[_HEAD_JOINT], axis=1) < head_radius
    lh = np.linalg.norm(verts - joints[_LEFT_WRIST], axis=1) < hand_radius
    rh = np.linalg.norm(verts - joints[_RIGHT_WRIST], axis=1) < hand_radius
    return head | lh | rh


def clean_displacement(
    D: np.ndarray,
    verts: np.ndarray,
    joints: np.ndarray,
    sym: np.ndarray,
    *,
    head_radius: float = DEFAULT_HEAD_RADIUS_M,
    hand_radius: float = DEFAULT_HAND_RADIUS_M,
) -> np.ndarray:
    """Symmetrize the displacement, then zero it on the head + hands."""
    D_clean = symmetrize_displacement(D, sym)
    mask = head_hand_mask(verts, joints, head_radius=head_radius,
                          hand_radius=hand_radius)
    D_clean = D_clean.copy()
    D_clean[mask] = 0.0
    return D_clean


def clean_fit_npz(
    fit_npz: Path,
    out_npz: Path,
    *,
    model_folder: str = "data/body_models",
    gender: str | None = None,
    num_betas: int = 300,
    pose_deg: float = DEFAULT_APOSE_DEG,
    head_radius: float = DEFAULT_HEAD_RADIUS_M,
    hand_radius: float = DEFAULT_HAND_RADIUS_M,
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
    bm = smplx.create(model_path=model_folder, model_type="smplx", gender=g,
                      num_betas=num_betas, use_pca=False, flat_hand_mean=True,
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
                                     head_radius=head_radius,
                                     hand_radius=hand_radius)
        if verbose:
            asym = np.abs(D - symmetrize_displacement(D, sym)).mean() * 1000
            nz = int(head_hand_mask(verts, joints, head_radius=head_radius,
                                    hand_radius=hand_radius).sum())
            print(f"  clean-fit: symmetrized D (mean asym {asym:.2f}mm), "
                  f"zeroed {nz} head/hand verts, canonical A-pose {pose_deg:.0f}deg")
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
