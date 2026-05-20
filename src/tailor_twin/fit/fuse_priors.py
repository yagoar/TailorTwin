"""Combine a LiDAR-chamfer fit with a SHAPY image-regression fit.

Both fits live in the same SMPL-X shape space. They capture different
signals:

  * LiDAR fit (300 betas)  — chamfer-fitted to the Stray-Scanner mesh;
    encodes the subject's personal high-frequency shape (limb thickness
    ratios, torso depth, asymmetries) at the cost of clothing/scan
    artefacts at body distance.

  * SHAPY fit (10 betas)   — CAESAR-trained image regressor; encodes
    real-anthropometry priors on macro shape modes (bust, waist, hip
    girths), clothing-invariant by training, but blind to the betas
    LiDAR's chamfer fit pushes into the 11..299 range.

Fusion rule: weighted average of the first ``n_shared`` betas (default
10 — SHAPY's whole shape vector), LiDAR's remaining betas[10:] kept
verbatim. The result is a single SMPL-X fit npz that flows into
``tailor-twin refine-tape`` → ``measure cli`` → viewer the same way as
either parent.

Run via ``tailor-twin fuse-priors``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FusionInputs:
    """Parsed metadata from the two fits we're about to fuse."""
    lidar_betas: np.ndarray   # (N_lidar,) float32, N_lidar = e.g. 300
    shapy_betas: np.ndarray   # (N_shapy,) float32, N_shapy = e.g. 10 + padding
    gender: str               # must match between parents (we check)
    displacement: np.ndarray | None
    n_betas_out: int


def _load_betas(path: Path) -> tuple[np.ndarray, str, np.ndarray | None]:
    """Read betas + gender + (optional) displacement from a fit npz."""
    d = np.load(path)
    if "betas" not in d.files:
        raise ValueError(f"fit npz missing 'betas': {path}")
    betas = d["betas"].astype(np.float64)
    if betas.ndim != 1:
        raise ValueError(
            f"unexpected betas shape {betas.shape} in {path}; expected 1-D")
    from .fit import fit_gender
    gender = fit_gender(d)
    disp = d["displacement"] if "displacement" in d.files else None
    return betas, gender, disp


def fuse_betas(
    lidar_npz: Path,
    shapy_npz: Path,
    *,
    shapy_weight: float = 0.5,
    n_shared: int = 10,
) -> tuple[np.ndarray, str, np.ndarray | None]:
    """Return the fused beta vector + gender + LiDAR's displacement.

    ``shapy_weight`` ∈ [0, 1]:
      0 → LiDAR-only (ignore SHAPY)
      1 → SHAPY-only on the shared block, LiDAR-only on the tail
      0.5 → equal weight (default)

    ``n_shared`` defaults to 10 (SHAPY's whole CAESAR-trained vector).
    Lowering it (e.g. 5) means LiDAR keeps more of the dominant shape
    modes; raising it past SHAPY's actual beta count clips at SHAPY's
    real beta count.
    """
    lidar_betas, lidar_gender, lidar_disp = _load_betas(lidar_npz)
    shapy_betas, shapy_gender, _ = _load_betas(shapy_npz)

    if lidar_gender != shapy_gender:
        raise ValueError(
            f"gender mismatch: lidar={lidar_gender!r} shapy={shapy_gender!r} "
            "— fuse must use the same SMPL-X model basis on both sides")

    n_shared = min(n_shared, lidar_betas.shape[0], shapy_betas.shape[0])

    fused = lidar_betas.copy()
    fused[:n_shared] = (
        (1.0 - shapy_weight) * lidar_betas[:n_shared] +
        shapy_weight * shapy_betas[:n_shared]
    )
    return fused.astype(np.float32), lidar_gender, lidar_disp


def fuse_to_fit_npz(
    lidar_npz: Path,
    shapy_npz: Path,
    *,
    out_npz: Path,
    shapy_weight: float = 0.5,
    n_shared: int = 10,
    a_pose_shoulder_deg: float = 30.0,
    model_folder: str = "data/body_models",
    keep_lidar_displacement: bool = False,
    verbose: bool = True,
) -> dict:
    """Fuse two fits and write the result as a tailor-twin fit npz.

    The saved fit is in canonical A-pose (default 30° shoulder
    rotation, T-pose at 0°) so it lines up with the refine-tape /
    SHAPY-import output convention. Pose, transl, global_orient are
    zeroed; vertices + joints recomputed by SMPL-X forward.

    ``keep_lidar_displacement`` carries the SMPL-X+D field through
    from the LiDAR fit. False by default — earlier experiments showed
    fused +D fights tape refinement (rigid local displacements clamp
    girths the optimizer wants to move)."""
    fused_betas, gender, lidar_disp = fuse_betas(
        lidar_npz, shapy_npz,
        shapy_weight=shapy_weight, n_shared=n_shared)

    n_betas = fused_betas.shape[0]

    import smplx
    import torch
    bm = smplx.create(
        model_path=model_folder, model_type="smplx",
        gender=gender, num_betas=n_betas,
        use_pca=False, flat_hand_mean=True, batch_size=1,
    )

    # Canonical pose: zero body pose with optional A-pose shoulders.
    from .refine_to_tape import _build_a_pose
    body_pose = _build_a_pose(a_pose_shoulder_deg).astype(np.float32)
    global_orient = np.zeros((3,), dtype=np.float32)
    transl = np.zeros((3,), dtype=np.float32)

    with torch.no_grad():
        out = bm(
            betas=torch.from_numpy(fused_betas[None, :]),
            body_pose=torch.from_numpy(body_pose.reshape(1, -1)),
            global_orient=torch.from_numpy(global_orient.reshape(1, -1)),
            transl=torch.from_numpy(transl.reshape(1, 3)),
            return_full_pose=False,
        )
    verts = out.vertices[0].cpu().numpy().astype(np.float32)
    joints = out.joints[0].cpu().numpy().astype(np.float32)

    if keep_lidar_displacement and lidar_disp is not None and \
       lidar_disp.shape == verts.shape:
        disp_out = lidar_disp.astype(np.float32)
        verts = verts + disp_out
    else:
        disp_out = np.zeros_like(verts)

    payload = {
        "betas": fused_betas,
        "body_pose": body_pose,
        "global_orient": global_orient,
        "transl": transl,
        "z": np.array([]),
        "smplx_vertices": verts,
        "smplx_joints": joints,
        "final_chamfer": np.array([float("nan")]),
        "displacement": disp_out,
        "gender": np.array(gender),
        "person_given_name": np.array(""),
        "person_family_name": np.array(""),
        "person_birth_date": np.array(""),
    }
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **payload)

    if verbose:
        print(f"wrote {out_npz}  "
              f"(gender={gender}, betas={n_betas}, "
              f"shapy_weight={shapy_weight}, n_shared={n_shared})")
    return payload
