"""SHAPY regressor output loader.

SHAPY (https://github.com/muelea/shapy) is a single-image SMPL-X body-shape
regressor trained on CAESAR. Each image produces one npz with:

    betas         (10,)        CAESAR shape betas
    body_pose     (21, 3, 3)   per-joint rotation matrices
    global_rot    (1, 3, 3)    pelvis rotation matrix
    transl        (3,)         camera-frame translation
    vertices      (10475, 3)   posed mesh in image/camera frame
    v_shaped      (10475, 3)   shape-only mesh in canonical T-pose
    joints        (123, 3)     SMPL-X joints
    faces         (20908, 3)   SMPL-X topology (matches tailor-twin's)
    measurements  dict         SHAPY's own body measurements
    camera        (3,)         scale + xy weak-perspective camera
    focal_length_in_px, focal_length_in_mm, sensor_width, center, shift_x/y

This module:
  - reads one or more SHAPY npz files
  - averages betas across views (multi-view fusion)
  - converts rotation matrices to axis-angle so the result lines up
    with tailor-twin's `fit_npz` schema (axis-angle, np.float32)
  - writes a canonical-pose fit_npz that the rest of the pipeline
    (measure CLI, refine-tape, viewer) consumes unchanged

Use case: photo front-end. Take 3-5 photos of the subject in A-pose,
run SHAPY's regressor on each, then feed all the resulting npzs to
``shapy_to_fit`` to land in tailor-twin's pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# SMPL-X topology constants (must match tailor-twin's body_models/smplx).
EXPECTED_VERTICES = 10475
EXPECTED_FACES = 20908
EXPECTED_BODY_POSE_JOINTS = 21


@dataclass
class ShapyResult:
    """Parsed + canonicalised output from one SHAPY npz."""
    betas: np.ndarray             # (10,) CAESAR shape betas
    body_pose_aa: np.ndarray      # (21, 3) axis-angle
    global_orient_aa: np.ndarray  # (3,) axis-angle
    transl: np.ndarray            # (3,)
    v_shaped: np.ndarray          # (10475, 3) canonical T-pose mesh
    joints: np.ndarray            # (J, 3) SMPL-X joints (full 123 from SHAPY)
    source_path: Path             # provenance


def _mat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3,3) -> axis-angle (3,). Uses scipy for stability."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec().astype(np.float32)


def load_shapy_npz(path: Path) -> ShapyResult:
    """Read one SHAPY regressor output npz and canonicalise it."""
    path = Path(path)
    d = np.load(path, allow_pickle=True)

    needed = ("betas", "body_pose", "global_rot", "transl", "v_shaped")
    missing = [k for k in needed if k not in d.files]
    if missing:
        raise ValueError(
            f"SHAPY npz missing keys {missing}: {path}\n"
            f"got: {list(d.files)}")

    betas = d["betas"].astype(np.float32)
    if betas.ndim != 1:
        raise ValueError(
            f"unexpected betas shape {betas.shape} in {path}; expected (N,)")

    body_pose_mat = d["body_pose"].astype(np.float32)  # (21, 3, 3)
    if body_pose_mat.shape != (EXPECTED_BODY_POSE_JOINTS, 3, 3):
        raise ValueError(
            f"unexpected body_pose shape {body_pose_mat.shape} in {path}; "
            f"expected ({EXPECTED_BODY_POSE_JOINTS}, 3, 3)")
    body_pose_aa = np.stack([_mat_to_axis_angle(body_pose_mat[i])
                              for i in range(EXPECTED_BODY_POSE_JOINTS)])

    global_mat = d["global_rot"].astype(np.float32)
    if global_mat.shape == (1, 3, 3):
        global_mat = global_mat[0]
    if global_mat.shape != (3, 3):
        raise ValueError(
            f"unexpected global_rot shape {global_mat.shape} in {path}")
    global_orient_aa = _mat_to_axis_angle(global_mat)

    transl = d["transl"].astype(np.float32)
    if transl.shape != (3,):
        raise ValueError(
            f"unexpected transl shape {transl.shape} in {path}")

    v_shaped = d["v_shaped"].astype(np.float32)
    if v_shaped.shape != (EXPECTED_VERTICES, 3):
        raise ValueError(
            f"unexpected v_shaped shape {v_shaped.shape} in {path}; "
            f"expected ({EXPECTED_VERTICES}, 3)")

    joints = (d["joints"].astype(np.float32)
              if "joints" in d.files else np.zeros((0, 3), dtype=np.float32))

    return ShapyResult(
        betas=betas,
        body_pose_aa=body_pose_aa,
        global_orient_aa=global_orient_aa,
        transl=transl,
        v_shaped=v_shaped,
        joints=joints,
        source_path=path,
    )


def fuse_multi_view(results: list[ShapyResult]) -> np.ndarray:
    """Average CAESAR betas across multiple views.

    Pose/translation differ per view (camera location). Shape is the
    pose-invariant per-subject signal — averaging across views is the
    standard SHAPY multi-view fusion (paper Table 4 reports a 1-2 cm
    measurement gain over single-view).

    Future: down-weight views with low detection confidence. SHAPY
    doesn't expose a scalar confidence in the saved npz, so the simple
    arithmetic mean is the current baseline.
    """
    if not results:
        raise ValueError("fuse_multi_view: need at least one ShapyResult")
    n_betas = results[0].betas.shape[0]
    for r in results:
        if r.betas.shape[0] != n_betas:
            raise ValueError(
                f"beta count mismatch: {r.source_path} has {r.betas.shape[0]} "
                f"betas, expected {n_betas}")
    return np.mean(np.stack([r.betas for r in results]), axis=0)


def shapy_to_fit(
    results: list[ShapyResult],
    *,
    out_num_betas: int = 300,
    canonical_pose: bool = True,
    gender: str = "female",
    model_folder: str = "data/body_models",
) -> dict:
    """Convert one-or-more SHAPY outputs into a tailor-twin fit_npz dict.

    Multi-view: betas are averaged; the first result's pose / transl
    is kept only when ``canonical_pose=False``. By default the saved
    fit is in canonical T-pose (body_pose / global_orient / transl
    zeroed) and ``smplx_vertices`` are recomputed by running SMPL-X
    forward — same convention as ``fit.refine_to_tape.save_refined_fit``.

    Pad CAESAR's 10 betas with zeros out to ``out_num_betas`` so the
    npz is compatible with the rest of the pipeline (measure CLI,
    refine-tape) that expects a fixed beta count matching the
    SMPL-X model build."""
    if not results:
        raise ValueError("shapy_to_fit: no SHAPY results supplied")

    betas_avg = fuse_multi_view(results)
    if betas_avg.shape[0] > out_num_betas:
        raise ValueError(
            f"SHAPY produced {betas_avg.shape[0]} betas but "
            f"out_num_betas={out_num_betas}; cannot truncate")
    betas_padded = np.zeros(out_num_betas, dtype=np.float32)
    betas_padded[: betas_avg.shape[0]] = betas_avg

    # Always recompute the mesh + joints in tailor-twin's SMPL-X model
    # space. SHAPY's v_shaped was rendered through SHAPY's own SMPL-X
    # instance (gender=neutral by default in their demos); we want
    # consistency with the gender + num_betas used downstream.
    import smplx
    import torch

    bm = smplx.create(
        model_path=model_folder, model_type="smplx",
        gender=gender, num_betas=out_num_betas,
        use_pca=False, flat_hand_mean=True, batch_size=1,
    )

    if canonical_pose:
        body_pose = np.zeros((EXPECTED_BODY_POSE_JOINTS, 3), dtype=np.float32)
        global_orient = np.zeros((3,), dtype=np.float32)
        transl = np.zeros((3,), dtype=np.float32)
    else:
        first = results[0]
        body_pose = first.body_pose_aa
        global_orient = first.global_orient_aa
        transl = first.transl

    with torch.no_grad():
        out = bm(
            betas=torch.from_numpy(betas_padded[None, :]),
            body_pose=torch.from_numpy(body_pose.reshape(1, -1)),
            global_orient=torch.from_numpy(global_orient.reshape(1, -1)),
            transl=torch.from_numpy(transl.reshape(1, 3)),
            return_full_pose=False,
        )
    verts = out.vertices[0].cpu().numpy().astype(np.float32)
    joints = out.joints[0].cpu().numpy().astype(np.float32)

    payload = {
        "betas": betas_padded,
        "body_pose": body_pose,
        "global_orient": global_orient,
        "transl": transl,
        "z": np.array([]),
        "smplx_vertices": verts,
        "smplx_joints": joints,
        "final_chamfer": np.array([float("nan")]),  # no chamfer fit here
        "displacement": np.zeros((EXPECTED_VERTICES, 3), dtype=np.float32),
        "gender": np.array(gender),
        # Person info: empty unless caller writes it in after.
        "person_given_name": np.array(""),
        "person_family_name": np.array(""),
        "person_birth_date": np.array(""),
    }
    return payload


def save_fit_payload(payload: dict, out_path: Path) -> None:
    """Write the dict from ``shapy_to_fit`` as a tailor-twin fit_npz."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
