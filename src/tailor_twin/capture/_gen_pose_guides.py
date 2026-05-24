"""Generate the pose-guide silhouettes for the capture webapp.

Renders a neutral SMPL-X body in the canonical A-pose and rasterizes its
front and side silhouettes to translucent PNGs. The capture page overlays
these inside the guide frame so the user can line their body up with the
expected pose and framing for each shot.

Run once (the PNGs are committed as static assets)::

    python -m tailor_twin.capture._gen_pose_guides

Outputs ``capture/static/pose_front.png`` and ``pose_side.png``.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _render_silhouette(
    verts: np.ndarray, faces: np.ndarray, axis_x: int, *,
    px_h: int = 1000, flip_x: bool = False,
    keep_verts: np.ndarray | None = None,
) -> np.ndarray:
    """Rasterize a filled body silhouette → RGBA uint8, cropped to the
    body, white at ~50% alpha on a transparent background.

    ``axis_x`` selects the horizontal world axis (0 = X → front view,
    2 = Z → side view); Y is always the vertical axis. ``keep_verts``,
    if given, is a per-vertex bool mask — only faces whose three
    vertices are all kept are drawn (used to drop the arms from the
    side guide so the hand does not poke out of the profile)."""
    if keep_verts is not None:
        faces = faces[keep_verts[faces].all(axis=1)]
    x = verts[:, axis_x].copy()
    y = verts[:, 1].copy()
    if flip_x:
        x = -x
    x0, x1 = x.min(), x.max()
    y0, y1 = y.min(), y.max()
    pad = 0.04 * (y1 - y0)
    scale = px_h / (y1 - y0 + 2 * pad)
    w = int(round((x1 - x0 + 2 * pad) * scale))
    h = px_h
    px = ((x - x0 + pad) * scale).astype(np.int32)
    py = ((y1 + pad - y) * scale).astype(np.int32)   # flip Y → image down

    mask = np.zeros((h, w), np.uint8)
    tri = np.stack([px[faces], py[faces]], axis=-1)  # (F,3,2)
    cv2.fillPoly(mask, tri, 255)
    # Per-triangle fill leaves pinholes where projected faces are thin.
    # Re-fill the external contour so the body is solid (a silhouette
    # has no genuine interior holes).
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(mask)
    if cnts:
        cv2.drawContours(solid, [max(cnts, key=cv2.contourArea)],
                         -1, 255, cv2.FILLED)
    mask = solid

    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = (mask > 0).astype(np.uint8) * 128
    return rgba


def main() -> int:
    import smplx
    import torch

    from ..fit.refine_to_tape import _build_a_pose

    model_folder = "data/body_models"
    bm = smplx.create(model_path=model_folder, model_type="smplx",
                      gender="neutral", num_betas=10, use_pca=False,
                      flat_hand_mean=True, batch_size=1)
    faces = np.asarray(bm.faces, dtype=np.int64)

    def verts_for(shoulder_deg: float) -> np.ndarray:
        pose = _build_a_pose(shoulder_deg).astype(np.float32)
        with torch.no_grad():
            out = bm(
                betas=torch.zeros(1, 10),
                body_pose=torch.from_numpy(pose.reshape(1, -1)),
                global_orient=torch.zeros(1, 3),
                transl=torch.zeros(1, 3),
            )
        return out.vertices[0].cpu().numpy().astype(np.float64)

    static = Path(__file__).parent / "static"
    static.mkdir(parents=True, exist_ok=True)

    # Front guide: arms moderately out (shoulder 58° below the T-pose
    # horizontal) — enough of an arm gap for segmentation, narrow enough
    # that the figure fits a portrait phone frame. Arms shown.
    front = _render_silhouette(verts_for(58.0), faces, axis_x=0)

    # Side guide: arms dropped entirely (torso + head + legs only) so the
    # hand does not poke out of the profile. flip_x makes the figure face
    # image-left → the subject's RIGHT side is toward the camera, the
    # same orientation as the reference side photo.
    from ..measure.regions import region_vertex_mask
    side_keep = region_vertex_mask(
        ("torso", "left_leg", "right_leg"),
        model_folder=model_folder, gender="neutral")
    side = _render_silhouette(verts_for(78.0), faces, axis_x=2,
                              flip_x=True, keep_verts=side_keep)

    # Paired-capture side guide: the subject poses ONCE and both phones
    # fire, so the side guide must show the SAME A-pose as the front
    # (58°), not the 4-shot arms-down pose. Arms are still dropped from
    # the render so the hand doesn't poke the profile — the subject
    # aligns the torso/legs and holds the arms out per the front phone.
    side_apose = _render_silhouette(verts_for(58.0), faces, axis_x=2,
                                    flip_x=True, keep_verts=side_keep)
    cv2.imwrite(str(static / "pose_front.png"), front)
    cv2.imwrite(str(static / "pose_side.png"), side)
    cv2.imwrite(str(static / "pose_side_apose.png"), side_apose)
    print(f"wrote {static/'pose_front.png'}  {front.shape}")
    print(f"wrote {static/'pose_side.png'}  {side.shape}")
    print(f"wrote {static/'pose_side_apose.png'}  {side_apose.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
