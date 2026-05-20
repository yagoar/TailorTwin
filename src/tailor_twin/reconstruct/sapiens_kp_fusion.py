"""Keypoint-driven multi-view fusion of Sapiens2 pointmaps.

Multi-view monocular pointmap fusion is hard because each frame's
output lives in its own camera coordinate frame and the only common
signal between views is the subject's actual body geometry. The trick
this module uses: read 2D body keypoints from Sapiens2's pose model,
look up their 3D positions in the matching pointmap, then derive a
per-frame "body frame" (midhip origin, head-up axis, hip-axis right,
front-back from the cross product). All points in a frame become
points in body coordinates — which automatically align across views.

Required inputs per frame
-------------------------

* ``<name>.ply`` — Sapiens2 pointmap, row-major H×W of (x, y, z) in
  camera frame. Background pixels included; we drop them by a depth
  band + optional Sapiens2-seg mask.
* ``<name>_keypoints.json`` — OpenPose BODY_25 JSON. Pulled out of
  ``sapiens_to_openpose.convert_predictions`` (Sapiens2 308-kpt mapped
  to OpenPose body subset).

If you have the Sapiens2 308-kpt predictions JSON instead, point this
module at it directly via ``load_sapiens_predictions``.

The output is one fused point cloud in body-frame coordinates:
+Y up, +X subject-right, +Z subject-front, origin at midhip.
Downstream chamfer-fit SMPL-X consumes this cloud unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


# COCO-WholeBody / Sapiens2 keypoint indices used for body-frame build.
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_HIP = 11
KP_R_HIP = 12
KP_NOSE = 0


@dataclass
class FrameKeypoints3D:
    """3D body landmarks looked up from a frame's pointmap."""
    l_shoulder: np.ndarray
    r_shoulder: np.ndarray
    l_hip: np.ndarray
    r_hip: np.ndarray
    nose: np.ndarray
    scores: dict[str, float]
    midhip: np.ndarray = None  # filled in post-init
    midshoulder: np.ndarray = None

    def __post_init__(self):
        self.midhip = (self.l_hip + self.r_hip) / 2.0
        self.midshoulder = (self.l_shoulder + self.r_shoulder) / 2.0

    @property
    def confidence(self) -> float:
        """Mean confidence across the keypoints used for body-frame build."""
        return float(np.mean([
            self.scores["l_shoulder"],
            self.scores["r_shoulder"],
            self.scores["l_hip"],
            self.scores["r_hip"],
        ]))


def _lookup_3d(
    points_flat: np.ndarray,  # (H*W, 3)
    H: int, W: int,
    u: float, v: float,
) -> np.ndarray:
    """Bilinear-ish lookup: nearest pixel for now (4-tap interp possible later)."""
    u_int = int(np.clip(round(u), 0, W - 1))
    v_int = int(np.clip(round(v), 0, H - 1))
    return points_flat[v_int * W + u_int].astype(np.float32)


def load_sapiens_predictions(path: Path) -> dict[str, dict]:
    """Parse Sapiens2 predictions JSON into {image_name: {keypoints, scores, bbox}}.

    Picks the highest-mean-body-score instance per frame (same heuristic
    as ``sapiens_to_openpose._pick_primary_instance``)."""
    data = json.loads(Path(path).read_text())
    out: dict[str, dict] = {}
    for frame in data.get("frames", []):
        name = frame["image_name"]
        instances = frame.get("instances", []) or []
        if not instances:
            continue
        best = max(
            instances,
            key=lambda inst: np.mean((inst.get("keypoint_scores") or [0])[5:17])
            if len(inst.get("keypoint_scores") or []) >= 17 else 0,
        )
        out[name] = best
    return out


def extract_3d_keypoints(
    ply_path: Path,
    instance: dict,
    *,
    image_h: int,
    image_w: int,
) -> FrameKeypoints3D | None:
    """Read the .ply (row-major H×W) and look up 3D positions for the
    six body landmarks. Returns None if any required keypoint has
    confidence < 0.3 — the resulting body frame would be unreliable."""
    kpts = instance.get("keypoints") or []
    scores = instance.get("keypoint_scores") or []
    if len(kpts) < 13 or len(scores) < 13:
        return None
    if min(scores[KP_L_SHOULDER], scores[KP_R_SHOULDER],
           scores[KP_L_HIP], scores[KP_R_HIP]) < 0.3:
        return None

    pc = o3d.io.read_point_cloud(str(ply_path))
    pts_flat = np.asarray(pc.points, dtype=np.float32)
    expected = image_h * image_w
    if pts_flat.shape[0] < expected:
        raise ValueError(
            f"ply {ply_path.name} has {pts_flat.shape[0]} points; "
            f"expected at least {expected} (H={image_h}, W={image_w})")

    def get(kp_idx: int) -> np.ndarray:
        u, v = kpts[kp_idx]
        return _lookup_3d(pts_flat, image_h, image_w, u, v)

    fk = FrameKeypoints3D(
        l_shoulder=get(KP_L_SHOULDER),
        r_shoulder=get(KP_R_SHOULDER),
        l_hip=get(KP_L_HIP),
        r_hip=get(KP_R_HIP),
        nose=get(KP_NOSE),
        scores={
            "l_shoulder": float(scores[KP_L_SHOULDER]),
            "r_shoulder": float(scores[KP_R_SHOULDER]),
            "l_hip": float(scores[KP_L_HIP]),
            "r_hip": float(scores[KP_R_HIP]),
            "nose": float(scores[KP_NOSE]),
        },
    )
    return fk


def body_frame(kp: FrameKeypoints3D) -> tuple[np.ndarray, np.ndarray]:
    """Construct an orthonormal body frame from 3D body landmarks.

    Returns (R, origin):
      R       — 3×3 rotation matrix mapping body-frame coords to camera-frame
                (columns = [right, up, forward] in camera frame)
      origin  — midhip in camera frame.

    To convert a camera-frame point P_cam to body frame:
        P_body = R.T @ (P_cam - origin)

    Conventions follow SMPL-X / tailor-twin's body model: +Y up, +X to
    the subject's right (camera's left when subject faces camera),
    +Z forward (out of the subject's chest).
    """
    # +Y up: midhip → midshoulder.
    up = kp.midshoulder - kp.midhip
    nrm = np.linalg.norm(up)
    if nrm < 1e-6:
        raise ValueError("degenerate up vector — midhip == midshoulder")
    up = up / nrm

    # +X to subject's right. Subject's right hip lives on subject's right
    # side, so right = (R_hip - L_hip).
    right_raw = kp.r_hip - kp.l_hip
    # Orthogonalise against up to keep frame strictly orthonormal.
    right = right_raw - np.dot(right_raw, up) * up
    nrm = np.linalg.norm(right)
    if nrm < 1e-6:
        raise ValueError("degenerate right vector — l_hip == r_hip after"
                          " projecting away up")
    right = right / nrm

    # +Z forward via right × up (right-handed). For a subject facing
    # the camera (frame 0), forward points into the camera's -Z, so the
    # nose should also lie on +Z in body frame as a sanity check.
    forward = np.cross(right, up)
    forward = forward / np.linalg.norm(forward)

    # Sign check: nose should have positive Z in body frame (in front
    # of the chest). If it's behind, flip the forward axis (and reflect
    # right to keep the frame right-handed).
    nose_body_z = np.dot(kp.nose - kp.midhip, forward)
    if nose_body_z < 0:
        forward = -forward
        right = -right

    R = np.column_stack([right, up, forward]).astype(np.float64)
    origin = kp.midhip.astype(np.float64)
    return R, origin


def points_to_body_frame(
    points_camera: np.ndarray,
    R: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    """Apply (R, origin) to map camera-frame points to body frame."""
    return ((points_camera.astype(np.float64) - origin) @ R).astype(np.float32)


# --------------------------------------------------------------------- #
# End-to-end fusion                                                     #
# --------------------------------------------------------------------- #


@dataclass
class FusionFrame:
    """Bookkeeping for one Sapiens2 frame entering fusion."""
    name: str
    points_camera: np.ndarray
    colors: np.ndarray | None
    keypoints_3d: FrameKeypoints3D


def _depth_band_filter(points: np.ndarray, band_m: float = 0.5) -> np.ndarray:
    z = points[:, 2]
    hist, edges = np.histogram(z, bins=200, range=(-4.0, 2.0))
    peak_z = (edges[hist.argmax()] + edges[hist.argmax() + 1]) * 0.5
    return np.abs(z - peak_z) <= band_m


def load_frame(
    ply_path: Path,
    predictions: dict[str, dict],
    *,
    image_h: int,
    image_w: int,
    image_name_for_kpts: str | None = None,
    target_points: int = 80_000,
    rng_seed: int = 0,
) -> FusionFrame | None:
    """Read a Sapiens2 .ply, look up its keypoints from ``predictions``,
    drop background points, downsample, return a ``FusionFrame``.

    Returns None if the frame's keypoints are unusable (low confidence
    or missing — usually means the subject was cropped or occluded)."""
    image_name = image_name_for_kpts or (ply_path.stem + ".jpg")
    inst = predictions.get(image_name)
    if inst is None:
        # Try without extension.
        inst = predictions.get(ply_path.stem)
    if inst is None:
        print(f"  {ply_path.name}: no predictions match — skipped")
        return None

    kp3d = extract_3d_keypoints(
        ply_path, inst, image_h=image_h, image_w=image_w)
    if kp3d is None:
        print(f"  {ply_path.name}: low-confidence keypoints — skipped")
        return None

    pc = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pc.points, dtype=np.float32)
    cols = (np.asarray(pc.colors, dtype=np.float32)
            if len(pc.colors) else None)

    keep = _depth_band_filter(pts)
    pts = pts[keep]
    if cols is not None:
        cols = cols[keep]

    if target_points and len(pts) > target_points:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(pts), target_points, replace=False)
        pts = pts[idx]
        if cols is not None:
            cols = cols[idx]

    return FusionFrame(
        name=ply_path.stem,
        points_camera=pts,
        colors=cols,
        keypoints_3d=kp3d,
    )


def fuse_kp(
    frames: list[FusionFrame],
    *,
    refine_with_icp: bool = True,
    icp_max_distance: float = 0.05,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Multi-view fuse by mapping every frame into its keypoint-derived
    body frame, then merging. Optional ICP refines local mis-alignment
    between consecutive frames' body clouds."""
    if not frames:
        raise ValueError("fuse_kp: no frames")

    merged_pts = np.empty((0, 3), dtype=np.float32)
    merged_cols = None if frames[0].colors is None else np.empty((0, 3), dtype=np.float32)
    merged_pc = o3d.geometry.PointCloud()

    for i, f in enumerate(frames):
        R, origin = body_frame(f.keypoints_3d)
        body_pts = points_to_body_frame(f.points_camera, R, origin)
        body_cols = f.colors

        if refine_with_icp and i > 0 and len(merged_pts) > 0:
            src_pc = o3d.geometry.PointCloud()
            src_pc.points = o3d.utility.Vector3dVector(body_pts.astype(np.float64))
            result = o3d.pipelines.registration.registration_icp(
                src_pc, merged_pc, icp_max_distance, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=40,
                    relative_fitness=1e-7,
                    relative_rmse=1e-7,
                ),
            )
            T = result.transformation
            body_pts = (T[:3, :3] @ body_pts.T).T + T[:3, 3]
            body_pts = body_pts.astype(np.float32)

        merged_pts = np.vstack([merged_pts, body_pts])
        if merged_cols is not None and body_cols is not None:
            merged_cols = np.vstack([merged_cols, body_cols])

        merged_pc = o3d.geometry.PointCloud()
        merged_pc.points = o3d.utility.Vector3dVector(merged_pts.astype(np.float64))

    return merged_pts, merged_cols
