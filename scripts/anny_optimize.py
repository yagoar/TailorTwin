"""Differentiable silhouette + tape fit using NAVER Anny body model.

Drop-in replacement experiment for ``silhouette_optimize.py`` — Anny's
interpretable phenotype params + measurement-named blendshapes replace
SMPL-X's 20 PCA betas. Tape girths (bust/waist/hip/etc.) get dedicated
``measure-*-circ-incr`` blendshapes so they don't fight for capacity in
a shared PCA budget like SMPL-X.

Optimizes:
* phenotype: height, weight, muscle, proportions (gender + age fixed)
* shape local_changes: ~16 named blendshapes covering torso girths,
  limb circumferences, breast/butt volume, torso v-shape, leg lengths.

Loss:
* soft silhouette IoU front + side (same rasterizer as SMPL-X path)
* tape girths sliced at landmark Y levels (bust/underbust/waist/
  highhip/hip/knee) — fed from landmarks JSON when given
* anthropometric height pulled to ``--height`` cm

Pose: canonical A-pose, no pose-opt for the spike. Add bone-pose
optimization once the shape fit is working.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import anny

from tailor_twin.fit.silhouette_render import soft_silhouette, _gauss_kernel


def _face_points(verts: torch.Tensor, faces: torch.Tensor,
                  bary: torch.Tensor) -> torch.Tensor:
    """Sample (B, 3) barycentric points per face. ``bary`` is (K, 3)
    with rows summing to 1 — see ``make_bary_grid``."""
    tri = verts[faces]              # (F, 3, 3)
    # einsum: 'kb, fbd -> fkd' → (F, K, 3) → flatten to (F*K, 3)
    pts = torch.einsum("kb,fbd->fkd", bary, tri)
    return pts.reshape(-1, 3)


def _cauchy_perimeter(pts2d: torch.Tensor, n_dirs: int = 64) -> torch.Tensor:
    """Convex perimeter of a 2D point cloud via Cauchy's formula
    (integral of width across angles). Differentiable through pts."""
    theta = torch.linspace(0, np.pi, n_dirs + 1, device=pts2d.device,
                            dtype=pts2d.dtype)[:-1]
    cos = torch.cos(theta); sin = torch.sin(theta)
    proj = pts2d[:, 0:1] * cos[None, :] + pts2d[:, 1:2] * sin[None, :]
    widths = proj.max(dim=0).values - proj.min(dim=0).values
    return widths.mean() * np.pi


def _slice_girth(pts: torch.Tensor, lo, hi, axis: int = 1) -> torch.Tensor:
    """Convex perimeter of pts in horizontal band lo <= pts[:, axis] <= hi."""
    coord = pts[:, axis]
    if isinstance(lo, torch.Tensor): lo = lo.detach()
    if isinstance(hi, torch.Tensor): hi = hi.detach()
    sel = (coord >= lo) & (coord <= hi)
    if int(sel.sum()) < 6:
        return pts.new_tensor(0.0)
    other = [a for a in (0, 1, 2) if a != axis]
    return _cauchy_perimeter(pts[sel][:, other])


torch.set_default_dtype(torch.float32)


# Body-shape local_changes we optimise. All in [-1, 1]; default 0 = mean.
SHAPE_LC = [
    "measure-bust-circ-incr",
    "measure-underbust-circ-incr",
    "measure-waist-circ-incr",
    "measure-hips-circ-incr",
    "measure-thigh-circ-incr",
    "measure-knee-circ-incr",
    "measure-upperarm-circ-incr",
    "measure-neck-circ-incr",
    "measure-shoulder-dist-incr",
    "measure-waisttohip-dist-incr",
    "measure-upperleg-height-incr",
    "measure-lowerleg-height-incr",
    "measure-napetowaist-dist-incr",
    "measure-frontchest-dist-incr",
    "breast-volume-vert-up",
    "buttocks-volume-incr",
    "torso-vshape-incr",
    "stomach-tone-incr",
]


def load_seg_silhouette(seg_path: str,
                         drop_classes: tuple[int, ...] = ()) -> np.ndarray:
    """Sapiens-style binary silhouette (float32 0/1). Same logic as
    silhouette_optimize.load_torso."""
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
    return mask.astype(np.float32)


def crop_to_body(mask: np.ndarray, target_h: int, target_w: int,
                  margin: float = 0.06) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    y0, y1 = ys.min(), ys.max(); x0, x1 = xs.min(), xs.max()
    mh = int((y1 - y0) * margin); mw = int((x1 - x0) * margin)
    y0 = max(0, y0 - mh); y1 = min(mask.shape[0], y1 + mh)
    x0 = max(0, x0 - mw); x1 = min(mask.shape[1], x1 + mw)
    crop = mask[y0:y1, x0:x1]
    ch, cw = crop.shape
    target_ar = target_w / target_h
    cur_ar = cw / ch
    if cur_ar < target_ar:
        new_w = int(round(ch * target_ar)); pad = (new_w - cw) // 2
        padded = np.zeros((ch, new_w), dtype=crop.dtype)
        padded[:, pad:pad + cw] = crop
    else:
        new_h = int(round(cw / target_ar)); pad = (new_h - ch) // 2
        padded = np.zeros((new_h, cw), dtype=crop.dtype)
        padded[pad:pad + ch, :] = crop
    return cv2.resize(padded, (target_w, target_h),
                      interpolation=cv2.INTER_AREA)


def make_bary_grid(subdiv: int = 3, device=None) -> torch.Tensor:
    n = subdiv
    b = [(i / n, j / n, 1 - i / n - j / n)
         for i in range(n + 1) for j in range(n + 1 - i)]
    return torch.tensor(b, dtype=torch.float32, device=device)


def project_view(verts: torch.Tensor, view: str, img_h: int, img_w: int,
                 scale: torch.Tensor, tx: torch.Tensor, ty: torch.Tensor
                 ) -> torch.Tensor:
    """Verts are already in Y-up frame (Z-up Anny was swapped)."""
    horiz = verts[:, 0] if view == "front" else verts[:, 2]
    y = verts[:, 1]
    u = img_w / 2 + scale * horiz + tx
    v = img_h / 2 - scale * y + ty
    return torch.stack([u, v], dim=-1)


def anny_to_yup(verts_zup: torch.Tensor) -> torch.Tensor:
    """Anny is Z-up; rotate to Y-up so existing projection code works."""
    return torch.stack([verts_zup[:, 0], verts_zup[:, 2], -verts_zup[:, 1]],
                        dim=-1)


def soft_iou_loss(pred: torch.Tensor, gt: torch.Tensor,
                   uncov_w: float = 15.0, over_w: float = 0.3) -> torch.Tensor:
    inter = (pred * gt).sum()
    union = (pred + gt - pred * gt).sum()
    iou = inter / (union + 1e-6)
    uncov = torch.relu(gt - pred).mean()
    over = torch.relu(pred - gt).mean()
    return (1.0 - iou) + uncov_w * uncov + over_w * over


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--front-seg", type=Path, required=True)
    ap.add_argument("--side-seg",  type=Path, required=True)
    ap.add_argument("--landmarks", type=Path, default=None,
                    help="landmark_editor JSON — supplies per-girth photo Y "
                         "and tape cm targets.")
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--height", type=float, default=160.0,
                    help="Subject height in cm.")
    ap.add_argument("--gender", default="female")
    ap.add_argument("--age", type=float, default=0.55,
                    help="Anny age in [0,1] (0=infant, 0.5≈25y, 1=elder).")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--img-h", type=int, default=384)
    ap.add_argument("--img-w", type=int, default=240)
    ap.add_argument("--gauss-sigma", type=float, default=0.7)
    args = ap.parse_args(argv)

    dev = torch.device("cpu")

    # Load Anny model with full local_changes.
    print("loading Anny model …")
    bm = anny.create_fullbody_model(local_changes="all").to(dev)
    anth = anny.Anthropometry(bm)
    faces_np = bm.get_triangular_faces().detach().cpu().numpy().astype(np.int64)
    n_quads = bm.faces.shape[0]
    # Body-part face segmentation: drop hands, eyes, mouth cavity, tongue.
    # Mask is over QUAD faces (each quad → 2 triangles).
    keep_labels = ["body", "head", "foot.L", "foot.R"]
    quad_keep = anny.get_face_segmentation_mask(bm, keep_labels).cpu().numpy()
    tri_keep_face = np.repeat(quad_keep, 2)
    # Strip arm verts (drop face if any vert is dominantly an arm bone) so
    # the T-pose mesh's outstretched arms don't fight the arm-stripped photo
    # silhouette during shape fit. Same trick as the SMPL-X path.
    bone_w = bm.vertex_bone_weights.detach().cpu().numpy()
    bone_i = bm.vertex_bone_indices.detach().cpu().numpy()
    dom_bone = bone_i[np.arange(bone_i.shape[0]),
                       bone_w.argmax(axis=1)]
    arm_keys = ("upperarm", "lowerarm", "wrist", "hand", "thumb",
                 "index", "middle", "ring", "pinky", "clavicle", "shoulder")
    arm_bone_mask = np.array([
        any(k in lbl.lower() for k in arm_keys)
        for lbl in bm.bone_labels
    ], dtype=bool)
    arm_v_mask = arm_bone_mask[dom_bone]
    no_arm_face_mask = ~arm_v_mask[faces_np].any(axis=1)
    # Full mesh (for side view): drop interior only.
    full_mask = tri_keep_face
    # Front (arm-stripped): drop interior + arms.
    front_mask_faces = tri_keep_face & no_arm_face_mask
    faces_full  = torch.from_numpy(
        faces_np[full_mask].astype(np.int64)).to(dev)
    faces_noarm = torch.from_numpy(
        faces_np[front_mask_faces].astype(np.int64)).to(dev)
    print(f"faces: total {len(faces_np)}  full-mesh {len(faces_full)}  "
          f"arm-stripped {len(faces_noarm)}")
    n_verts = bm.template_vertices.shape[0]
    print(f"Anny: {n_verts} verts, {len(faces_np)} faces, "
          f"{bm.bone_count} bones")

    # Photo silhouettes (drop arm classes like the SMPL-X path).
    DROP_FRONT = (6, 7, 11, 15, 20); DROP_SIDE = (1,)
    front_full = load_seg_silhouette(str(args.front_seg), DROP_FRONT)
    side_full  = load_seg_silhouette(str(args.side_seg),  DROP_SIDE)
    front_mask = crop_to_body(front_full, args.img_h, args.img_w)
    side_mask  = crop_to_body(side_full,  args.img_h, args.img_w)
    front_gt = torch.from_numpy(front_mask.copy().astype(np.float32)).to(dev)
    side_gt  = torch.from_numpy(side_mask.copy().astype(np.float32)).to(dev)

    # Tape targets + landmark Y from JSON.
    tape_targets: dict[str, float] = {}      # name -> cm
    lm_y_frac: dict[str, float] = {}         # name -> frac from head_top (0..1)
    if args.landmarks and args.landmarks.exists():
        d = json.loads(args.landmarks.read_text())
        for name, by in (d.get("lines_y") or {}).items():
            if not by: continue
            ys_s = np.where(side_full > 0)[0]
            s_top, s_h = ys_s.min(), ys_s.max() - ys_s.min()
            ys_f = np.where(front_full > 0)[0]
            f_top, f_h = ys_f.min(), ys_f.max() - ys_f.min()
            if by.get("side") is not None:
                lm_y_frac[name] = (by["side"] - s_top) / s_h
            elif by.get("front") is not None:
                lm_y_frac[name] = (by["front"] - f_top) / f_h
        mm = d.get("measurements") or {}
        for k in ("bust", "underbust", "waist", "highhip", "hip",
                  "neck", "highbust"):
            if k in mm and mm[k]:
                tape_targets[k] = float(mm[k])
        if "knee_circ" in mm and mm["knee_circ"]:
            tape_targets["knee"] = float(mm["knee_circ"])
        if "bicep" in mm and mm["bicep"]:
            tape_targets["bicep"] = float(mm["bicep"])
        print(f"loaded tape targets: {tape_targets}")
        print(f"loaded lm_y_frac:    {lm_y_frac}")

    # Optimizable parameters.
    pheno_init = {"height": 0.36, "weight": 0.4, "muscle": 0.5,
                  "proportions": 0.5}
    pheno = {k: torch.nn.Parameter(torch.tensor(v, device=dev))
             for k, v in pheno_init.items()}
    gender_val = 0.0 if args.gender == "female" else 1.0
    fixed = {"gender": torch.tensor(gender_val, device=dev),
             "age": torch.tensor(float(args.age), device=dev)}
    lc = {k: torch.nn.Parameter(torch.tensor(0.0, device=dev)) for k in SHAPE_LC}

    # Camera: orthographic, anchored to photo body bbox.
    body_h_m = args.height / 100.0
    s_init = args.img_h * 0.88 / body_h_m
    scale = torch.nn.Parameter(torch.tensor(s_init, device=dev))

    # Photo body-bbox centres (for translation auto-align per iter).
    def bbox_yc(m):
        ys = np.where(m > 0)[0]; return float((ys.min() + ys.max()) / 2)
    def bbox_xc(m):
        xs = np.where(m > 0)[1]; return float((xs.min() + xs.max()) / 2)
    photo_xc_f = bbox_xc(front_mask); photo_yc_f = bbox_yc(front_mask)
    photo_xc_s = bbox_xc(side_mask);  photo_yc_s = bbox_yc(side_mask)
    photo_xc_f_t = torch.tensor(photo_xc_f, device=dev)
    photo_yc_f_t = torch.tensor(photo_yc_f, device=dev)
    photo_xc_s_t = torch.tensor(photo_xc_s, device=dev)
    photo_yc_s_t = torch.tensor(photo_yc_s, device=dev)

    bary = make_bary_grid(3, dev)
    kernel = _gauss_kernel(args.gauss_sigma, dev)

    opt = torch.optim.Adam(
        [{"params": list(pheno.values()), "lr": args.lr},
         {"params": list(lc.values()),    "lr": args.lr * 1.5},
         {"params": [scale],              "lr": args.lr * 5.0}])

    def aligned_translate(pts2d, xc, yc):
        x_mid = (pts2d[:, 0].max() + pts2d[:, 0].min()) / 2
        y_mid = (pts2d[:, 1].max() + pts2d[:, 1].min()) / 2
        return torch.stack([pts2d[:, 0] + xc - x_mid,
                            pts2d[:, 1] + yc - y_mid], dim=-1)

    def forward():
        out = bm(phenotype_kwargs={**fixed, **pheno},
                 local_changes_kwargs=lc)
        verts_zup = out["vertices"][0].to(torch.float32)
        verts = anny_to_yup(verts_zup)
        # Rescale so anthropometric height matches user input.
        anth_h_m = anth.height(out["vertices"])[0].to(torch.float32)
        verts = verts * (body_h_m / (anth_h_m + 1e-9))
        pts_f = _face_points(verts, faces_noarm, bary)
        uv_f = project_view(pts_f, "front", args.img_h, args.img_w,
                            scale, torch.zeros(()),
                            torch.zeros(()))
        uv_f = aligned_translate(uv_f, photo_xc_f_t, photo_yc_f_t)
        sil_f = soft_silhouette(uv_f, args.img_h, args.img_w, kernel)
        pts_s = _face_points(verts, faces_full, bary)
        uv_s = project_view(pts_s, "side", args.img_h, args.img_w,
                            scale, torch.zeros(()),
                            torch.zeros(()))
        uv_s = aligned_translate(uv_s, photo_xc_s_t, photo_yc_s_t)
        sil_s = soft_silhouette(uv_s, args.img_h, args.img_w, kernel)
        return verts, sil_f, sil_s, out

    print(f"\noptimizing {args.iters} iters …")
    best_loss = float("inf"); best_state = None
    for it in range(args.iters):
        opt.zero_grad()
        verts, sil_f, sil_s, out = forward()
        loss_f = soft_iou_loss(sil_f, front_gt)
        loss_s = soft_iou_loss(sil_s, side_gt)

        # Tape girths: slice mesh at landmark Y level.
        tape_loss = verts.new_zeros(())
        if tape_targets and lm_y_frac:
            y = verts[:, 1]; y_lo, y_hi = y.min(), y.max(); span = y_hi - y_lo
            for name, cm in tape_targets.items():
                lm_name = {"bust": "bust", "underbust": "underbust",
                            "waist": "waist", "highhip": "highhip",
                            "hip": "hip", "knee": "knee"}.get(name)
                if lm_name is None or lm_name not in lm_y_frac:
                    continue
                frac = lm_y_frac[lm_name]
                lvl = y_lo + (1.0 - frac) * span
                g_m = _slice_girth(verts, lvl - 0.02, lvl + 0.02)
                g_cm = g_m * 100.0
                tape_loss = tape_loss + ((g_cm - cm) / 10.0) ** 2

        # Height prior — strong pull on anthropometric height.
        anth_h_cm = anth.height(out["vertices"])[0].to(torch.float32) * 100.0
        reg_h = 0.5 * (anth_h_cm - args.height) ** 2

        # Light reg on phenotype + lc params to keep them in plausible range.
        reg_p = 0.5 * sum((p - 0.5) ** 2 for p in pheno.values())
        reg_l = 0.1 * sum((c) ** 2 for c in lc.values())

        # Downweight tape so silhouette dominates initially.
        loss = loss_f + loss_s + 0.1 * tape_loss + reg_h + reg_p + reg_l
        loss.backward()
        opt.step()
        # Clamp phenotypes to [0, 1] and local_changes to [-1.5, 1.5]
        with torch.no_grad():
            for p in pheno.values(): p.clamp_(0.0, 1.0)
            for c in lc.values(): c.clamp_(-1.5, 1.5)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {"pheno": {k: v.detach().clone() for k, v in pheno.items()},
                          "lc": {k: v.detach().clone() for k, v in lc.items()},
                          "scale": scale.detach().clone(), "iter": it,
                          "iou_f": 1 - loss_f.item(),
                          "iou_s": 1 - loss_s.item()}
        if it % 25 == 0 or it == args.iters - 1:
            verts_h_cm = anth_h_cm.item()
            print(f"  iter {it:4d}  total={loss.item():.3f}  "
                  f"IoU_f={1-loss_f.item():.3f}  IoU_s={1-loss_s.item():.3f}  "
                  f"H={verts_h_cm:.1f}cm")

    print(f"\nbest at iter {best_state['iter']}: "
          f"IoU_f={best_state['iou_f']:.3f} IoU_s={best_state['iou_s']:.3f}")
    for k, v in best_state["pheno"].items(): pheno[k].data = v
    for k, v in best_state["lc"].items(): lc[k].data = v
    scale.data = best_state["scale"]

    # Save fit + render targets
    with torch.no_grad():
        verts, sil_f, sil_s, out = forward()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_npz = args.out_prefix.with_name(args.out_prefix.name + "_anny_fit.npz")
    np.savez(out_npz,
             phenotype={k: float(v) for k, v in pheno.items()},
             local_changes={k: float(v) for k, v in lc.items()},
             gender=args.gender, age=args.age, height_cm=args.height,
             vertices=verts.detach().cpu().numpy().astype(np.float32),
             faces=faces_np.astype(np.int32))
    print(f"wrote {out_npz}")
    cv2.imwrite(str(args.out_prefix.with_name(args.out_prefix.name +
                                              "_target_front.png")),
                (front_mask * 255).astype(np.uint8))
    cv2.imwrite(str(args.out_prefix.with_name(args.out_prefix.name +
                                              "_target_side.png")),
                (side_mask * 255).astype(np.uint8))
    cv2.imwrite(str(args.out_prefix.with_name(args.out_prefix.name +
                                              "_render_front.png")),
                (sil_f.detach().cpu().numpy() * 255).astype(np.uint8))
    cv2.imwrite(str(args.out_prefix.with_name(args.out_prefix.name +
                                              "_render_side.png")),
                (sil_s.detach().cpu().numpy() * 255).astype(np.uint8))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
