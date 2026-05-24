"""Render full SMPL-X wireframe over Sapiens silhouette.

Produces the lm3/lm4-style overlay: every mesh triangle drawn in yellow
on top of the photo silhouette (gray). Red where the photo silhouette
extends past the mesh; pale yellow where the mesh extends past the
silhouette. Full mesh (arms included) over full silhouette (no arm
class drop) — diagnostic view of the fit, not the optimizer's loss view.

Usage:
  python scripts/silhouette_overlay.py \
    --fit data/results/pair_20260522_090554_lm5_smplx_fit.npz \
    --front-seg data/results/pair_20260522_090554_sapiens/out/front_seg.npy \
    --side-seg  data/results/pair_20260522_090554_sapiens/out/side_seg.npy \
    --out-prefix data/results/pair_20260522_090554_lm5 \
    --height 160 --gender female
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import smplx

from silhouette_optimize import load_torso, project_view
from tailor_twin.fit.refine_to_tape import _build_a_pose


def crop_aspect_free(mask: np.ndarray, target_h: int,
                      margin: float = 0.06) -> np.ndarray:
    """Crop body bbox + margin, resize to ``target_h``, width preserves
    body aspect (no padding). Returns the mask at (target_h, target_w)."""
    ys, xs = np.where(mask > 0)
    y0, y1 = ys.min(), ys.max(); x0, x1 = xs.min(), xs.max()
    mh = int((y1 - y0) * margin); mw = int((x1 - x0) * margin)
    y0 = max(0, y0 - mh); y1 = min(mask.shape[0], y1 + mh)
    x0 = max(0, x0 - mw); x1 = min(mask.shape[1], x1 + mw)
    crop = mask[y0:y1, x0:x1]
    ch, cw = crop.shape
    target_w = int(round(cw * target_h / ch))
    return cv2.resize(crop, (target_w, target_h),
                       interpolation=cv2.INTER_AREA)


# Colours (BGR for OpenCV)
COL_SIL_ONLY = (0,   0,   220)      # photo silhouette, no mesh = RED
COL_SIL_BG   = (200, 200, 200)      # photo silhouette = light gray
COL_WIRE     = (0,   230, 230)      # mesh edges = yellow


def render_wire_over_sil(verts: np.ndarray, faces: np.ndarray,
                          view: str, sil_gray: np.ndarray,
                          img_h: int, img_w: int, scale: float,
                          photo_xc: float, photo_yc: float) -> np.ndarray:
    """Project mesh, draw yellow wireframe over photo silhouette.

    Photo silhouette = gray. Mesh = yellow triangle edges (unique edges
    only). Red = photo silhouette pixels not covered by mesh (= mesh too
    narrow/short there).
    """
    v = torch.from_numpy(verts.astype(np.float32))
    s = torch.tensor(float(scale))
    zero = torch.tensor(0.0)
    uv = project_view(v, view, img_h, img_w, s, zero, zero).numpy()
    x_mid = (uv[:, 0].max() + uv[:, 0].min()) / 2
    uv[:, 0] += photo_xc - x_mid
    # Anchor mesh bottom (heel/sole when feet are posed flat — see
    # main() which lifts canonical toe-down by rotating the ankles)
    # to the photo silhouette bottom.
    photo_y_bot = float(np.where(sil_gray > 0.5)[0].max())
    mesh_y_bot = float(uv[:, 1].max())
    uv[:, 1] += photo_y_bot - mesh_y_bot

    # Mesh fill mask — used only to mark red (sil ∖ mesh)
    mesh_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for tri in faces:
        pts = uv[tri].astype(np.int32)
        cv2.fillConvexPoly(mesh_mask, pts, 1)
    mesh = mesh_mask > 0

    rgb = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    sil = sil_gray > 0.5
    rgb[sil]            = COL_SIL_BG
    rgb[sil & ~mesh]    = COL_SIL_ONLY

    # Unique edges only (each shared edge drawn once → 1/2 the lines)
    edges = np.concatenate([faces[:, [0, 1]],
                             faces[:, [1, 2]],
                             faces[:, [2, 0]]], axis=0)
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    uv_int = uv.astype(np.int32)
    for a, b in edges:
        cv2.line(rgb, tuple(uv_int[a]), tuple(uv_int[b]),
                  COL_WIRE, thickness=1, lineType=cv2.LINE_8)
    return rgb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--front-seg", type=Path, required=True)
    ap.add_argument("--side-seg",  type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--height", type=float, default=160.0)
    ap.add_argument("--gender", default="female")
    ap.add_argument("--img-h",  type=int, default=1600,
                    help="Render height; taller = finer wireframe. Width "
                         "is derived per-view from the body aspect.")
    ap.add_argument("--shoulder-deg", type=float, default=55.0,
                    help="A-pose shoulder drop angle (0 = T-pose, "
                         "55 = clear A-pose).")
    ap.add_argument("--keep-pose", action="store_true",
                    help="Render in fitted pose instead of canonical A-pose.")
    args = ap.parse_args(argv)

    d = np.load(args.fit, allow_pickle=True)
    betas = d["betas"].astype(np.float32)
    num_betas = len(betas)
    if args.keep_pose:
        body_pose = d["body_pose"].astype(np.float32).reshape(1, 63)
    else:
        # Canonical A-pose: shoulders dropped, head/neck/spine neutral so
        # the overlay shows body proportions in a clean reference pose
        # — independent of the photo's arm/head orientation.
        body_pose = _build_a_pose(args.shoulder_deg).astype(np.float32)
        body_pose = body_pose.reshape(1, 63)
    # Lift toes (ankle dorsiflexion) so the foot is flat. Canonical SMPL-X
    # has toes pointing forward+down — toe tip ~2cm below the heel — so
    # mesh y_min lands below the actual sole and the overlay's heel
    # appears 2cm below the photo's floor row. Rotate L_Ankle (body_pose
    # joint index 6) and R_Ankle (index 7) around the lateral X axis by
    # ~6° to bring the sole flat to ground.
    pose_view = body_pose.reshape(21, 3)
    pose_view[6, 0] += 0.10   # L_Ankle dorsiflex
    pose_view[7, 0] += 0.10   # R_Ankle dorsiflex
    body_pose = pose_view.reshape(1, 63)
    bm = smplx.create(model_path="data/body_models", model_type="smplx",
                       gender=args.gender, num_betas=num_betas,
                       use_pca=False, flat_hand_mean=True, batch_size=1)
    with torch.no_grad():
        out = bm(betas=torch.from_numpy(betas)[None],
                  body_pose=torch.from_numpy(body_pose),
                  global_orient=torch.zeros(1, 3),
                  transl=torch.zeros(1, 3))
    verts = out.vertices[0].numpy()
    faces = bm.faces.astype(np.int64)

    # Full silhouette — keep arms (diagnostic view, not optimizer loss view).
    front_full = load_torso(str(args.front_seg))
    side_full  = load_torso(str(args.side_seg))
    front_mask = crop_aspect_free(front_full, args.img_h)
    side_mask  = crop_aspect_free(side_full,  args.img_h)

    # Per-view scale: anchor mesh body extent to photo body bbox in the
    # cropped image. Uses MESH actual height (not user-input height) so
    # the rendered mesh fills the photo silhouette 1:1 vertically — no
    # scale-formula vs crop-margin mismatch, no shrink from height drift.
    mesh_h_m = float(verts[:, 1].max() - verts[:, 1].min())

    def bbox_yc(m):
        ys = np.where(m > 0)[0]; return float((ys.min() + ys.max()) / 2)
    def bbox_xc(m):
        xs = np.where(m > 0)[1]; return float((xs.min() + xs.max()) / 2)
    def torso_axis_x(m):
        """Median X of body center per row in upper-torso band — ignores
        outstretched arms / asymmetric stance so the front overlay lines
        up on the body axis, not the arm-span bbox center."""
        ys, _ = np.where(m > 0); y0, y1 = ys.min(), ys.max(); h = y1 - y0
        centers = []
        for y in range(y0 + int(0.15*h), y0 + int(0.55*h)):
            r = m[y]
            if r.any():
                xs_r = np.where(r > 0)[0]
                centers.append((xs_r.min() + xs_r.max()) / 2)
        return float(np.median(centers)) if centers else bbox_xc(m)
    def body_h_px(m):
        ys = np.where(m > 0)[0]; return float(ys.max() - ys.min())
    fxc, fyc = torso_axis_x(front_mask), bbox_yc(front_mask)
    sxc, syc = bbox_xc(side_mask),       bbox_yc(side_mask)
    scale_f = body_h_px(front_mask) / mesh_h_m
    scale_s = body_h_px(side_mask)  / mesh_h_m

    fh, fw = front_mask.shape
    sh, sw = side_mask.shape
    f_img = render_wire_over_sil(verts, faces, "front", front_mask,
                                  fh, fw, scale_f, fxc, fyc)
    s_img = render_wire_over_sil(verts, faces, "side",  side_mask,
                                  sh, sw, scale_s, sxc, syc)
    f_png = args.out_prefix.with_name(args.out_prefix.name + "_wire_front.png")
    s_png = args.out_prefix.with_name(args.out_prefix.name + "_wire_side.png")
    cv2.imwrite(str(f_png), f_img)
    cv2.imwrite(str(s_png), s_img)
    print(f"wrote {f_png}\nwrote {s_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
