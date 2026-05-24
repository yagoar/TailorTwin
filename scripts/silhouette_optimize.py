"""Differentiable silhouette optimization: betas+pose+camera vs front/side masks.

Reuses the soft rasterizer in ``tailor_twin.bmnet.abs`` to backprop a
silhouette-IoU loss into SMPL-X betas, body pose (shoulders/elbows/wrists),
and per-view scale + translation. Initialized from an existing fit npz.

Usage:
  python scripts/silhouette_optimize.py \
      --base-fit data/results/pair_20260522_090554_v3_smplx_fit.npz \
      --front-seg data/results/pair_20260522_090554_sapiens/out/front_seg.npy \
      --side-seg  data/results/pair_20260522_090554_sapiens/out/side_seg.npy \
      --out-prefix data/results/pair_20260522_090554_silhopt \
      --height 160 --gender female --iters 400
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import cv2
import torch
import smplx

from tailor_twin.bmnet.abs import (
    soft_silhouette, _face_points, _gauss_kernel, _slice_girth,
    _TORSO_LEVEL,
)
from tailor_twin.fit.silhouette import load_silhouette
from tailor_twin.measure.regions import region_vertex_mask


# Per-view Sapiens arm-class sets to DROP. Everything else (incl small
# foot/sock/shoe classes) is kept. Identified visually for this capture:
# front lateral arm classes = {6,7,11,15,20}; side has one lone arm
# blob {1} (hand sticking forward/back).
# Hair stays in silhouette so body-bbox Y reference (= top of head) is
# consistent with user-input height (top of head → heels).
DROP_CLASSES_FRONT = {6, 7, 11, 15, 20}
DROP_CLASSES_SIDE  = {1}
# Legacy KEEP-set (unused now, retained as docs)
KEEP_CLASSES_FRONT = None
KEEP_CLASSES_SIDE  = None


def load_torso(seg_path: str, drop_classes=(),
               trim_shoulder_cap: bool = False) -> np.ndarray:
    """Return binary silhouette = all non-background pixels minus
    ``drop_classes`` (arm Sapiens class IDs). Largest CC.

    ``trim_shoulder_cap`` legacy heuristic kept for backwards compat."""
    seg = np.load(seg_path)
    if seg.ndim == 3:
        seg = (seg.argmax(0) if seg.shape[0] < seg.shape[-1]
               else seg.argmax(-1))
    mask = seg > 0
    for c in drop_classes:
        mask &= seg != c
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n > 1:
        best = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        mask = lab == best
    mask = mask.astype(np.uint8)

    if trim_shoulder_cap and mask.any():
        # legacy fallback — only used if class-based strip insufficient
        ys, xs = np.where(mask > 0)
        y_top, y_bot = ys.min(), ys.max()
        body_h = y_bot - y_top + 1
        y_chest = y_top + int(0.30 * body_h)
        widths = []
        for y in range(max(y_top, y_chest - int(0.03 * body_h)),
                       y_chest + int(0.03 * body_h)):
            r = mask[y]
            if r.any():
                ws = np.where(r > 0)[0]
                widths.append(ws.max() - ws.min() + 1)
        chest_w = int(np.median(widths)) if widths else 1
        body_xc = (xs.min() + xs.max()) / 2
        x_lo = int(body_xc - chest_w / 2)
        x_hi = int(body_xc + chest_w / 2)
        new_mask = mask.copy()
        new_mask[y_top:y_chest, :x_lo] = 0
        new_mask[y_top:y_chest, x_hi:] = 0
        mask = new_mask
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n > 1:
            best = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            mask = (lab == best).astype(np.uint8)

    return mask.astype(np.float32)


def crop_to_body(mask: np.ndarray, target_h: int, target_w: int,
                 margin: float = 0.06) -> np.ndarray:
    """Crop tight bbox + margin, pad to target aspect (NO stretch), then
    resize to (target_h, target_w)."""
    ys, xs = np.where(mask > 0)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    mh = int((y1 - y0) * margin); mw = int((x1 - x0) * margin)
    y0 = max(0, y0 - mh); y1 = min(mask.shape[0], y1 + mh)
    x0 = max(0, x0 - mw); x1 = min(mask.shape[1], x1 + mw)
    crop = mask[y0:y1, x0:x1]
    ch, cw = crop.shape
    target_ar = target_w / target_h  # width/height
    cur_ar = cw / ch
    if cur_ar < target_ar:
        # Pad width
        new_w = int(round(ch * target_ar))
        pad = (new_w - cw) // 2
        padded = np.zeros((ch, new_w), dtype=crop.dtype)
        padded[:, pad:pad + cw] = crop
    else:
        # Pad height
        new_h = int(round(cw / target_ar))
        pad = (new_h - ch) // 2
        padded = np.zeros((new_h, cw), dtype=crop.dtype)
        padded[pad:pad + ch, :] = crop
    return cv2.resize(padded, (target_w, target_h),
                       interpolation=cv2.INTER_AREA)


def make_bary_grid(subdiv: int = 3) -> torch.Tensor:
    """Barycentric sample points per face — dense surface coverage."""
    n = subdiv
    b = [(i / n, j / n, 1 - i / n - j / n)
         for i in range(n + 1) for j in range(n + 1 - i)]
    return torch.tensor(b, dtype=torch.float32)


def project_view(verts: torch.Tensor, view: str, img_h: int, img_w: int,
                 scale: torch.Tensor, tx: torch.Tensor, ty: torch.Tensor
                 ) -> torch.Tensor:
    """Project verts to image coords for one view (front=X,Y or side=Z,Y)."""
    horiz = verts[:, 0] if view == "front" else verts[:, 2]
    y = verts[:, 1]
    u = img_w / 2 + scale * horiz + tx
    v = img_h / 2 - scale * y + ty
    return torch.stack([u, v], dim=-1)


def soft_iou_loss(pred_soft: torch.Tensor, gt_binary: torch.Tensor,
                  uncov_weight: float = 15.0,
                  overshoot_weight: float = 0.3,
                  shoulder_band_weight: float = 10.0,
                  bust_band_weight: float = 12.0,
                  waist_band_weight: float = 12.0,
                  bands: dict = None) -> torch.Tensor:
    """1 - IoU + L1 uncov/overshoot + extra push in bust + waist bands.

    Body image Y bands (0 = top of body, 1 = bottom).
    Image rows: small index = head/top, large index = feet/bottom.
    """
    inter = (pred_soft * gt_binary).sum()
    union = (pred_soft + gt_binary - pred_soft * gt_binary).sum()
    iou = inter / (union + 1e-6)
    uncov     = torch.relu(gt_binary - pred_soft).mean()
    overshoot = torch.relu(pred_soft - gt_binary).mean()
    H, W = gt_binary.shape
    rows_with_body = gt_binary.sum(dim=1) > 0
    idx = torch.where(rows_with_body)[0]
    shoulder_extra = pred_soft.new_zeros(())
    bust_extra = pred_soft.new_zeros(())
    waist_extra = pred_soft.new_zeros(())
    if len(idx) > 4:
        y_top = idx.min().item(); y_bot = idx.max().item()
        bh = y_bot - y_top
        def band_loss(lo, hi):
            bt = int(y_top + lo * bh); bb = int(y_top + hi * bh)
            if bb <= bt + 1: return pred_soft.new_zeros(())
            band = pred_soft.new_zeros(H); band[bt:bb] = 1.0
            return (torch.relu(gt_binary - pred_soft) * band[:, None]).mean()
        # Y bands as (lo, hi) fractions from top of body. Defaults if no
        # per-subject landmarks provided. Y is fraction from HEAD (0)
        # to FEET (1).
        defaults = {
            "shoulder": (0.10, 0.22),
            "bust":     (0.27, 0.38),
            "waist":    (0.40, 0.50),
        }
        b = bands if bands else defaults
        shoulder_extra = band_loss(*b.get("shoulder", defaults["shoulder"]))
        bust_extra     = band_loss(*b.get("bust",     defaults["bust"]))
        waist_extra    = band_loss(*b.get("waist",    defaults["waist"]))
    return ((1.0 - iou) + uncov_weight * uncov
            + overshoot_weight * overshoot
            + shoulder_band_weight * shoulder_extra
            + bust_band_weight * bust_extra
            + waist_band_weight * waist_extra)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-fit", type=Path, required=True)
    ap.add_argument("--front-seg", type=Path, required=True)
    ap.add_argument("--side-seg",  type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--height", type=float, default=160.0,
                    help="Target body height in cm — used only for scaling output.")
    ap.add_argument("--gender", default="female")
    ap.add_argument("--num-betas", type=int, default=10)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr-betas", type=float, default=0.02)
    ap.add_argument("--lr-pose",  type=float, default=0.005)
    ap.add_argument("--lr-cam",   type=float, default=2.0)
    ap.add_argument("--img-h",    type=int, default=384)
    ap.add_argument("--img-w",    type=int, default=288)
    ap.add_argument("--gauss-sigma", type=float, default=2.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pose-opt", action="store_true",
                    help="Also optimize arm pose joints (shoulder/elbow).")
    ap.add_argument("--target-bust",    type=float, default=None)
    ap.add_argument("--target-waist",   type=float, default=None)
    ap.add_argument("--target-highhip", type=float, default=None)
    ap.add_argument("--target-hip",     type=float, default=None)
    ap.add_argument("--target-neck",      type=float, default=None)
    ap.add_argument("--target-underbust", type=float, default=None)
    ap.add_argument("--target-thigh",     type=float, default=None)
    ap.add_argument("--target-knee",      type=float, default=None,
                    help="M05 knee circumference (cm).")
    ap.add_argument("--landmarks", type=Path, default=None,
                    help="JSON from scripts/landmark_editor.py — per-subject "
                         "Y positions for shoulder/bust/waist/hip used to "
                         "override hardcoded Y band ranges.")
    ap.add_argument("--pose-json", type=Path, default=None,
                    help="Sapiens pose_predictions.json — adds 2D joint "
                         "reprojection loss so body pose stays anatomical.")
    ap.add_argument("--pose-weight", type=float, default=0.5,
                    help="Weight of 2D joint reprojection loss.")
    ap.add_argument("--tape-weight",  type=float, default=2.0,
                    help="Weight on tape girth residuals (cm² coefficient).")
    ap.add_argument("--side-weight",  type=float, default=1.0,
                    help="Weight on side silhouette IoU loss. Set 0 to "
                         "rely on tape-derived depth instead.")
    ap.add_argument("--depth-from-tape", action="store_true",
                    help="Replace side silhouette IoU with per-Y depth "
                         "target derived from tape circumference + front "
                         "width via ellipse approximation.")
    args = ap.parse_args(argv)

    dev = torch.device(args.device)

    # --- Load init fit ---
    fit = np.load(args.base_fit, allow_pickle=True)
    base_betas = fit["betas"].astype(np.float32)
    betas_init = np.zeros(args.num_betas, dtype=np.float32)
    n_copy = min(len(base_betas), args.num_betas)
    betas_init[:n_copy] = base_betas[:n_copy]
    body_pose_init = fit["body_pose"].astype(np.float32).reshape(21, 3)
    print(f"loaded base fit: {len(base_betas)} betas → using {args.num_betas} "
          f"(first {n_copy} copied, rest zero)")

    # --- SMPL-X model + faces + bary grid ---
    bm = smplx.create(model_path="data/body_models", model_type="smplx",
                      gender=args.gender, num_betas=args.num_betas,
                      use_pca=False, flat_hand_mean=True, batch_size=1).to(dev)
    faces_full_np = bm.faces.astype(np.int64)
    W = bm.lbs_weights.detach().cpu().numpy()
    dom_joint = W.argmax(1)
    # Strip shoulder/elbow/wrist/fingers AND collars (16,17). The collar
    # strip costs a bit of upper-chest mesh but prevents shoulder bulge
    # spilling into front-view loss. Net positive on alignment.
    # Note: Sapiens arm classes drop slightly into the body shoulder edge,
    # so anatomical mesh shoulder is wider than what's left in photo →
    # mesh wider than photo silhouette at shoulder = ugly stubs. We choose
    # narrow mesh (thin red strip on outer shoulder) over stubs.
    arm_joints = set(range(16, 22)) | set(range(22, 55))
    arm_v_mask = np.isin(dom_joint, list(arm_joints))
    no_arm_face_mask = ~arm_v_mask[faces_full_np].any(axis=1)
    faces_full = torch.from_numpy(faces_full_np).to(dev)
    faces_noarm = torch.from_numpy(faces_full_np[no_arm_face_mask]).to(dev)
    print(f"faces: full {len(faces_full_np)}; arm-stripped {len(faces_noarm)}")
    bary = make_bary_grid(subdiv=3).to(dev)
    kernel = _gauss_kernel(args.gauss_sigma, dev)

    # Torso mask for differentiable girth measurement
    torso_mask_np = region_vertex_mask(("torso",),
                                       model_folder="data/body_models",
                                       gender=args.gender)
    torso_mask = torch.from_numpy(torso_mask_np).to(dev)

    # --- Photo masks (arm-stripped, cropped, resized) ---
    front_full = load_torso(str(args.front_seg), DROP_CLASSES_FRONT)
    side_full  = load_torso(str(args.side_seg),  DROP_CLASSES_SIDE)

    # Per-subject landmarks (v2 schema). Y fractions from head TOP to feet
    # (full body bbox, hair-top included → matches user-input height which
    # is top-of-head to heels).
    lm_y = {}                 # name -> y_frac_from_head (0 = head top, 1 = feet)
    lm_measurements = {}      # name -> cm (ground truth tape)
    if args.landmarks and args.landmarks.exists():
        lm_data = json.loads(args.landmarks.read_text())
        ys = np.where(side_full > 0)[0]
        sy0, sy1 = ys.min(), ys.max(); sbh = sy1 - sy0
        ysf = np.where(front_full > 0)[0]
        fy0, fy1 = ysf.min(), ysf.max(); fbh = fy1 - fy0
        # New schema lines_y
        for name, by_view in (lm_data.get("lines_y", {}) or {}).items():
            if not by_view: continue
            # Prefer side Y (cleaner profile), fall back to front
            if by_view.get("side") is not None:
                lm_y[name] = float((by_view["side"] - sy0) / sbh)
            elif by_view.get("front") is not None:
                lm_y[name] = float((by_view["front"] - fy0) / fbh)
        # Measurements override --target-* if provided
        for name, cm in (lm_data.get("measurements", {}) or {}).items():
            if cm is not None: lm_measurements[name] = float(cm)
        # Legacy v1 fallback (points keyed by view)
        if not lm_y and "side" in lm_data:
            for name, xy in lm_data.get("side", {}).items():
                if xy and len(xy) == 2:
                    lm_y[name] = float((xy[1] - sy0) / sbh)
        if lm_y:
            print(f"loaded landmarks (Y from head): "
                  f"{ {k: round(v,3) for k,v in lm_y.items()} }")
        if lm_measurements:
            print(f"loaded measurements: {lm_measurements}")
            # Override --target-* unless explicit
            if args.target_bust    is None and "bust"    in lm_measurements:
                args.target_bust    = lm_measurements["bust"]
            if args.target_waist   is None and "waist"   in lm_measurements:
                args.target_waist   = lm_measurements["waist"]
            if args.target_highhip is None and "highhip" in lm_measurements:
                args.target_highhip = lm_measurements["highhip"]
            if args.target_hip     is None and "hip"     in lm_measurements:
                args.target_hip     = lm_measurements["hip"]
            # NOTE: neck/underbust DISABLED by default. Their LEVELS Y
            # values (anatomy frac from feet) slice through head/upper-
            # chest where torso_mask still includes head verts → mesh
            # girth at that slice = head circ, not neck circ. Optimizer
            # then fights to shrink the head and destabilizes (lm5 best
            # at iter 41, then divergence). Pass --target-neck explicitly
            # only when LEVELS["neck"] points at a real neck-only slice.
            if args.target_thigh is None and "thigh" in lm_measurements:
                args.target_thigh = lm_measurements["thigh"]
            if args.target_knee is None and "knee_circ" in lm_measurements:
                args.target_knee = lm_measurements["knee_circ"]
    front_mask = crop_to_body(front_full, args.img_h, args.img_w)
    side_mask  = crop_to_body(side_full,  args.img_h, args.img_w)
    print(f"photo masks: front {front_full.shape} -> {front_mask.shape};"
          f" side {side_full.shape} -> {side_mask.shape}")
    cv2.imwrite(str(args.out_prefix.with_name(
        args.out_prefix.name + "_target_front.png")),
        (front_mask*255).astype(np.uint8))
    cv2.imwrite(str(args.out_prefix.with_name(
        args.out_prefix.name + "_target_side.png")),
        (side_mask*255).astype(np.uint8))

    # IMPORTANT: copy mask before tensor conversion so later jaw-masking
    # doesn't corrupt the numpy view used for bbox computation.
    front_gt = torch.from_numpy(front_mask.copy()).to(dev)
    side_gt  = torch.from_numpy(side_mask.copy()).to(dev)


    # --- Sapiens 2D pose keypoints (COCO body subset) ---
    # COCO body idx → SMPL-X joint idx mapping (left/right preserved)
    # COCO: 5=Lshoulder 6=Rshoulder 7=Lelbow 8=Relbow 9=Lwrist 10=Rwrist
    #       11=Lhip 12=Rhip 13=Lknee 14=Rknee 15=Lankle 16=Rankle
    # SMPL-X: 16=Lshoulder 17=Rshoulder 18=Lelbow 19=Relbow 20=Lwrist 21=Rwrist
    #         1=Lhip 2=Rhip 4=Lknee 5=Rknee 7=Lankle 8=Rankle
    COCO_TO_SMPLX = {5:16, 6:17, 7:18, 8:19, 9:20, 10:21,
                     11:1, 12:2, 13:4, 14:5, 15:7, 16:8}
    pose_kp = {"front": None, "side": None}  # (N,3) np.array x,y,score in full-img coords
    if args.pose_json and args.pose_json.exists():
        pj = json.loads(args.pose_json.read_text())
        for frame in pj.get("frames", []):
            name = frame.get("image_name", "")
            view = "front" if "front" in name.lower() else (
                   "side" if "side" in name.lower() else None)
            if not view: continue
            inst = frame.get("instances", [])
            if not inst: continue
            kps = np.array(inst[0]["keypoints"], dtype=np.float32)
            scs = np.array(inst[0]["keypoint_scores"], dtype=np.float32)
            pose_kp[view] = np.concatenate([kps, scs[:, None]], axis=1)
            print(f"pose: {view} {len(kps)} keypoints loaded")
    # Map full-image coords → crop coords for the optimizer
    def map_to_crop(kps_full, sil_full, crop_h, crop_w, margin=0.06):
        if kps_full is None: return None
        ys, xs = np.where(sil_full > 0)
        y0, y1 = ys.min(), ys.max(); x0, x1 = xs.min(), xs.max()
        mh = int((y1-y0)*margin); mw = int((x1-x0)*margin)
        y0 = max(0, y0-mh); y1 = min(sil_full.shape[0], y1+mh)
        x0 = max(0, x0-mw); x1 = min(sil_full.shape[1], x1+mw)
        ch, cw = y1-y0, x1-x0
        target_ar = crop_w / crop_h
        cur_ar = cw / ch
        if cur_ar < target_ar:
            new_w = int(round(ch * target_ar)); pad = (new_w - cw) // 2
            x0_eff = x0 - pad; cw = new_w
        else:
            new_h = int(round(cw / target_ar)); pad = (new_h - ch) // 2
            y0_eff = y0 - pad; ch = new_h
            x0_eff = x0
        if cur_ar < target_ar: y0_eff = y0
        sx = crop_w / cw; sy = crop_h / ch
        out = kps_full.copy()
        out[:, 0] = (out[:, 0] - x0_eff) * sx
        out[:, 1] = (out[:, 1] - y0_eff) * sy
        return out

    pose_kp_crop = {
        "front": map_to_crop(pose_kp["front"], front_full, args.img_h, args.img_w),
        "side":  map_to_crop(pose_kp["side"],  side_full,  args.img_h, args.img_w),
    }
    pose_kp_t = {}
    for v, kp in pose_kp_crop.items():
        if kp is not None:
            pose_kp_t[v] = torch.from_numpy(kp).to(dev)

    # Photo body anatomical centers in the cropped/resized images. Used
    # to deterministically align mesh per iteration.
    # Front: TORSO AXIS X (median of body center per row in upper-torso
    # band) — ignores asymmetric arm/leg stance. Y from bbox.
    # Side: bbox center (already symmetric Z-wise).
    def torso_axis_x(m):
        ys, _ = np.where(m > 0); y0, y1 = ys.min(), ys.max()
        h = y1 - y0
        centers = []
        for y in range(y0 + int(0.15*h), y0 + int(0.55*h)):
            r = m[y]
            if r.any():
                xs_r = np.where(r > 0)[0]
                centers.append((xs_r.min() + xs_r.max()) / 2)
        if not centers:
            xs = np.where(m > 0)[1]
            return (xs.min() + xs.max()) / 2.0
        return float(np.median(centers))
    def bbox_yc(m):
        ys = np.where(m > 0)[0]
        return float((ys.min() + ys.max()) / 2)
    def bbox_xc(m):
        xs = np.where(m > 0)[1]
        return float((xs.min() + xs.max()) / 2)
    photo_xc_f = torso_axis_x(front_mask)
    photo_yc_f = bbox_yc(front_mask)
    photo_xc_s = bbox_xc(side_mask)
    photo_yc_s = bbox_yc(side_mask)
    print(f"photo center: front torso-axis x={photo_xc_f:.1f} y={photo_yc_f:.0f}; "
          f"side bbox=({photo_xc_s:.0f},{photo_yc_s:.0f})")
    photo_xc_f_t = torch.tensor(photo_xc_f, device=dev)
    photo_yc_f_t = torch.tensor(photo_yc_f, device=dev)
    photo_xc_s_t = torch.tensor(photo_xc_s, device=dev)
    photo_yc_s_t = torch.tensor(photo_yc_s, device=dev)

    # Precompute per-landmark target depths from tape + front-photo widths.
    # Map _TORSO_LEVEL (mesh frac of body span) -> photo front silhouette
    # row, read width in cm, compute target depth = 2*C/π - W.
    depth_targets = {}  # name -> (mesh y-frac, target_depth_cm)
    if args.depth_from_tape:
        fys, fxs = np.where(front_full > 0)
        f_y_top = fys.min()
        f_h_px = fys.max() - fys.min() + 1
        f_cm_per_px = args.height / f_h_px
        for name, tape in (("chest", args.target_bust),
                            ("waist", args.target_waist),
                            ("hip",   args.target_hip)):
            if tape is None: continue
            yf = _TORSO_LEVEL[name]    # frac up from feet
            row_y = int(f_y_top + (1.0 - yf) * f_h_px)
            row = front_full[row_y]
            if not row.any(): continue
            xs_r = np.where(row > 0)[0]
            W_cm = (xs_r.max() - xs_r.min() + 1) * f_cm_per_px
            D_cm = max(2.0 * tape / np.pi - W_cm, 0.5 * W_cm)
            depth_targets[name] = (yf, D_cm)
            print(f"depth target {name}: y_frac={yf:.2f}  "
                  f"front_W={W_cm:.1f}cm  tape={tape:.0f}cm  "
                  f"-> target_D={D_cm:.1f}cm")

    # --- Optim variables ---
    betas = torch.nn.Parameter(torch.from_numpy(betas_init).to(dev))
    body_pose = torch.nn.Parameter(
        torch.from_numpy(body_pose_init.flatten()).to(dev))
    body_pose.requires_grad_(args.pose_opt)

    # Per-view camera: scale + xy translation (orthographic)
    # FIX scale from photo body height in px vs known body height in m.
    # photo body height fills crop_h × (1 - 2*margin); = crop_h * 0.88
    # so scale = (crop_h * 0.88) / body_height_m  with body_height_m = height_cm/100
    body_h_m = args.height / 100.0
    s_init = args.img_h * 0.88 / body_h_m
    # Scale per view is OPTIMIZED but held close to init via prior.
    scale_f = torch.nn.Parameter(torch.tensor(s_init, device=dev))
    scale_s = torch.nn.Parameter(torch.tensor(s_init, device=dev))
    print(f"init scale: {s_init:.1f} px/m  (body {body_h_m:.2f}m, "
          f"each view scales independently with prior anchor)")

    tx_f = torch.nn.Parameter(torch.zeros((), device=dev))
    ty_f = torch.nn.Parameter(torch.zeros((), device=dev))
    tx_s = torch.nn.Parameter(torch.zeros((), device=dev))
    ty_s = torch.nn.Parameter(torch.zeros((), device=dev))

    pose_params = [body_pose] if args.pose_opt else []
    opt = torch.optim.Adam([
        {"params": [betas], "lr": args.lr_betas},
        {"params": pose_params, "lr": args.lr_pose},
        {"params": [scale_f, scale_s], "lr": args.lr_cam * 0.5},
        {"params": [tx_f, ty_f, tx_s, ty_s], "lr": args.lr_cam},
    ])
    scale_init_t = torch.tensor(s_init, device=dev)

    def aligned_translate(pts2d, photo_xc, photo_yc):
        """Auto-translate projected 2D points so their bbox center
        equals (photo_xc, photo_yc). Differentiable through bbox of pts."""
        x_mid = (pts2d[:, 0].max() + pts2d[:, 0].min()) / 2
        y_mid = (pts2d[:, 1].max() + pts2d[:, 1].min()) / 2
        dx = photo_xc - x_mid
        dy = photo_yc - y_mid
        return torch.stack([pts2d[:, 0] + dx, pts2d[:, 1] + dy], dim=-1)

    # Build landmark-driven bands + tape levels if landmarks loaded.
    custom_bands = None
    custom_levels = None
    if lm_y:
        b = {}
        # Shoulder band: tight around shoulder Y
        if "shoulder" in lm_y:
            sy = lm_y["shoulder"]
            b["shoulder"] = (max(0.0, sy - 0.03), min(1.0, sy + 0.04))
        # Bust band: bust (line) → underbust
        if "bust" in lm_y and "underbust" in lm_y:
            b["bust"] = (max(0.0, lm_y["bust"] - 0.02),
                          min(1.0, lm_y["underbust"] + 0.02))
        elif "bust" in lm_y:
            b["bust"] = (max(0.0, lm_y["bust"] - 0.03),
                          min(1.0, lm_y["bust"] + 0.06))
        if "waist" in lm_y:
            b["waist"] = (max(0.0, lm_y["waist"] - 0.03),
                          min(1.0, lm_y["waist"] + 0.03))
        if b: custom_bands = b; print(f"landmark bands: {b}")
        # Tape Y levels (mesh space: 0=feet, 1=head)
        lv = {}
        if "neck"      in lm_y: lv["neck"]      = 1.0 - lm_y["neck"]
        if "bust"      in lm_y: lv["chest"]     = 1.0 - lm_y["bust"]
        if "underbust" in lm_y: lv["underbust"] = 1.0 - lm_y["underbust"]
        if "waist"     in lm_y: lv["waist"]     = 1.0 - lm_y["waist"]
        if "highhip"   in lm_y: lv["highhip"]   = 1.0 - lm_y["highhip"]
        if "hip"       in lm_y: lv["hip"]       = 1.0 - lm_y["hip"]
        if "knee"      in lm_y: lv["knee"]      = 1.0 - lm_y["knee"]
        if lv: custom_levels = lv; print(f"landmark tape levels: {lv}")

    def forward():
        out = bm(
            betas=betas[None],
            body_pose=body_pose[None],
            global_orient=torch.zeros(1, 3, device=dev),
            transl=torch.zeros(1, 3, device=dev),
        )
        verts = out.vertices[0]
        joints = out.joints[0]
        return _do_proj(verts, joints)

    def _do_proj(verts, joints):
        # Front: arm-stripped mesh vs arm-stripped photo, auto-aligned
        pts_f = _face_points(verts, faces_noarm, bary)
        uv_f = project_view(pts_f, "front", args.img_h, args.img_w,
                             scale_f, tx_f * 0, ty_f * 0)
        uv_f = aligned_translate(uv_f, photo_xc_f_t, photo_yc_f_t)
        sil_f = soft_silhouette(uv_f, args.img_h, args.img_w, kernel)
        # Side: FULL mesh, auto-aligned to photo body center
        pts_s = _face_points(verts, faces_full, bary)
        uv_s = project_view(pts_s, "side",  args.img_h, args.img_w,
                             scale_s, tx_s * 0, ty_s * 0)
        uv_s = aligned_translate(uv_s, photo_xc_s_t, photo_yc_s_t)
        sil_s = soft_silhouette(uv_s, args.img_h, args.img_w, kernel)

        # Project SMPL-X joints to 2D in same crop space (for pose loss)
        # Use mesh body bbox alignment same as silhouette
        proj_kp = {}
        for view, used in (("front", faces_noarm), ("side", faces_full)):
            pts_used = _face_points(verts, used, bary)
            if view == "front":
                uv_used = project_view(pts_used, "front", args.img_h, args.img_w,
                                       scale_f, tx_f*0, ty_f*0)
                ph_xc, ph_yc = photo_xc_f_t, photo_yc_f_t
                proj_v = joints[:, [0, 1]]
                scale = scale_f
            else:
                uv_used = project_view(pts_used, "side", args.img_h, args.img_w,
                                       scale_s, tx_s*0, ty_s*0)
                ph_xc, ph_yc = photo_xc_s_t, photo_yc_s_t
                proj_v = joints[:, [2, 1]]
                scale = scale_s
            mx_u = uv_used[:, 0]; my_u = uv_used[:, 1]
            mxc = (mx_u.max() + mx_u.min()) / 2
            myc = (my_u.max() + my_u.min()) / 2
            jx = proj_v[:, 0]*scale + (args.img_w/2) + (ph_xc - mxc)
            jy = -proj_v[:, 1]*scale + (args.img_h/2) + (ph_yc - myc)
            proj_kp[view] = torch.stack([jx, jy], dim=-1)
        return verts, sil_f, sil_s, proj_kp

    print(f"\noptimizing {args.iters} iters (pose_opt={args.pose_opt}) ...")
    best_loss = float("inf"); best_state = None
    for it in range(args.iters):
        opt.zero_grad()
        verts, sil_f, sil_s, proj_kp = forward()
        loss_f = soft_iou_loss(sil_f, front_gt, bands=custom_bands)
        loss_s_raw = soft_iou_loss(sil_s, side_gt, bands=custom_bands)
        loss_s = args.side_weight * loss_s_raw
        # Sapiens 2D joint reprojection loss
        pose_loss = verts.new_zeros(())
        if pose_kp_t:
            for view, kp in pose_kp_t.items():
                if kp is None: continue
                pj = proj_kp[view]
                for coco_idx, smplx_idx in COCO_TO_SMPLX.items():
                    if coco_idx >= len(kp): continue
                    target = kp[coco_idx]
                    if target[2] < 0.5: continue   # low confidence
                    diff = pj[smplx_idx] - target[:2]
                    pose_loss = pose_loss + (diff ** 2).sum() * target[2]
            # Normalize by image diag to make weight comparable to IoU loss
            pose_loss = pose_loss / (args.img_w * args.img_h)
        # Soft beta regularizer — quadratic + tiny quartic clamp.
        reg_betas = 0.001 * (betas ** 2).sum() + 1e-5 * (betas ** 6).sum()
        # Height prior: penalize departure from target body height
        verts_h = verts[:, 1].max() - verts[:, 1].min()
        reg_h = 20.0 * (verts_h - body_h_m) ** 2
        # Scale prior: TIGHT — lock scales near init so mesh can't shrink
        # to compensate (forces betas to do the work).
        reg_scale = 0.05 * ((scale_f - scale_init_t) ** 2
                            + (scale_s - scale_init_t) ** 2)
        # Tape girth loss (cm² penalty). Differentiable via _slice_girth.
        tape_loss = verts.new_zeros(())
        # highhip Y level chosen so mesh circ at this level ≈ measure-CLI
        # G08 reading (which lives at ~58% of body span from feet).
        LEVELS = {"neck":      0.85,
                  "chest":     _TORSO_LEVEL["chest"],
                  "underbust": 0.66,
                  "waist":     _TORSO_LEVEL["waist"],
                  "highhip":   0.58,
                  "hip":       _TORSO_LEVEL["hip"],
                  "knee":      0.30}
        if custom_levels:
            LEVELS.update(custom_levels)
        targets = (("neck",      args.target_neck),
                   ("chest",     args.target_bust),
                   ("underbust", args.target_underbust),
                   ("waist",     args.target_waist),
                   ("highhip",   args.target_highhip),
                   ("hip",       args.target_hip),
                   ("knee",      args.target_knee))
        if any(t is not None for _, t in targets):
            y = verts[:, 1]; y_lo, y_hi = y.min(), y.max(); span = y_hi - y_lo
            # Torso-mask-based girths (neck through hip).
            # Knee/thigh skipped — would need leg mask + perp-to-bone slice.
            TORSO_TAPES = {"neck", "chest", "underbust", "waist",
                           "highhip", "hip"}
            for name, tgt in targets:
                if tgt is None or tgt <= 0: continue
                if name not in TORSO_TAPES:
                    continue  # legs handled separately if at all
                lvl = y_lo + LEVELS[name] * span
                g_m = _slice_girth(verts[torso_mask],
                                    lvl - 0.02, lvl + 0.02)
                g_cm = g_m * 100.0
                tape_loss = tape_loss + ((g_cm - tgt) / 10.0) ** 2
            tape_loss = args.tape_weight * tape_loss
        # Tape-derived depth loss (per-landmark Z extent of torso band)
        depth_loss = verts.new_zeros(())
        if depth_targets:
            y = verts[:, 1]; y_lo, y_hi = y.min(), y.max(); span = y_hi - y_lo
            for name, (yf, D_cm) in depth_targets.items():
                lvl = y_lo + yf * span
                sel = (verts[:, 1] >= lvl - 0.02) & (verts[:, 1] <= lvl + 0.02)
                if int(sel.sum()) < 6: continue
                z = verts[sel, 2]
                D_mesh_cm = (z.max() - z.min()) * 100.0
                depth_loss = depth_loss + ((D_mesh_cm - D_cm) / 10.0) ** 2
            depth_loss = args.tape_weight * depth_loss
        loss = (loss_f + loss_s + reg_betas + reg_h + reg_scale
                + tape_loss + depth_loss + args.pose_weight * pose_loss)
        loss.backward()
        opt.step()
        cur_iou = (1 - loss_f.item() - loss_s.item())  # sum
        cur_combined = loss_f.item() + loss_s.item()
        if cur_combined < best_loss:
            best_loss = cur_combined
            best_state = {
                "betas": betas.detach().clone(),
                "body_pose": body_pose.detach().clone(),
                "tx_f": tx_f.detach().clone(), "ty_f": ty_f.detach().clone(),
                "tx_s": tx_s.detach().clone(), "ty_s": ty_s.detach().clone(),
                "iter": it,
                "iou_f": 1-loss_f.item(), "iou_s": 1-loss_s.item(),
            }
        if it % 25 == 0 or it == args.iters - 1:
            verts_h_cm = (verts[:,1].max() - verts[:,1].min()).item() * 100
            bn = betas.detach().norm().item()
            print(f"  iter {it:4d}  total={loss.item():.4f}  "
                  f"IoU_f={1-loss_f.item():.3f}  "
                  f"IoU_s={1-loss_s.item():.3f}  "
                  f"H={verts_h_cm:.1f}cm  |β|={bn:.2f}  "
                  f"sf={scale_f.item():.0f} ss={scale_s.item():.0f}")
    # Restore best
    print(f"\nbest at iter {best_state['iter']}: "
          f"IoU_f={best_state['iou_f']:.3f} IoU_s={best_state['iou_s']:.3f}")
    betas.data = best_state["betas"]; body_pose.data = best_state["body_pose"]
    tx_f.data = best_state["tx_f"]; ty_f.data = best_state["ty_f"]
    tx_s.data = best_state["tx_s"]; ty_s.data = best_state["ty_s"]

    # --- Save final fit npz ---
    with torch.no_grad():
        out = bm(
            betas=betas[None],
            body_pose=body_pose[None],
            global_orient=torch.zeros(1, 3, device=dev),
            transl=torch.zeros(1, 3, device=dev),
        )
        verts_final = out.vertices[0].cpu().numpy().astype(np.float32)
        joints_final = out.joints[0].cpu().numpy().astype(np.float32)

    # Pad betas to original count (300 if base had 300)
    betas_pad = np.zeros(args.num_betas, dtype=np.float32)
    betas_pad[: args.num_betas] = betas.detach().cpu().numpy()

    out_npz = args.out_prefix.with_name(
        args.out_prefix.name + "_smplx_fit.npz")
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: fit[k] for k in fit.files}
    payload["betas"] = betas_pad
    payload["body_pose"] = body_pose.detach().cpu().numpy().reshape(21, 3)
    payload["smplx_vertices"] = verts_final
    payload["smplx_joints"] = joints_final
    np.savez(out_npz, **payload)
    print(f"\nwrote {out_npz}")

    # Save final rendered silhouettes for inspection
    with torch.no_grad():
        _, sil_f, sil_s, _ = forward()
        cv2.imwrite(str(args.out_prefix.with_name(
            args.out_prefix.name + "_render_front.png")),
            (sil_f.cpu().numpy() * 255).astype(np.uint8))
        cv2.imwrite(str(args.out_prefix.with_name(
            args.out_prefix.name + "_render_side.png")),
            (sil_s.cpu().numpy() * 255).astype(np.uint8))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
