"""Fit SMPL-X pose to Sapiens 2D keypoints (orthographic SMPLify-lite).

The silhouette width-transfer pipeline needs the SMPL-X body posed the
way the subject actually stood — arms out in the front shot, arms down
in the side shot — so a horizontal mesh slice lines up with the same
anatomical level in the photo. Guessing a canonical A-pose mis-aligns
every slice.

Sapiens' 308-keypoint pose detector gives reliable 2-D body joints. This
module fits SMPL-X ``global_orient`` + ``body_pose`` and an orthographic
camera (scale + 2-D translation) so the projected SMPL-X joints land on
those keypoints. Shape betas are held fixed — pose only. A weak L2 pose
prior keeps unconstrained joints (spine) near neutral.

The camera is orthographic because the capture app holds the phone
vertical and the subject fills the frame — the same near-orthographic
assumption the silhouette fit already relies on.
"""
from __future__ import annotations

import numpy as np

# COCO-17 body keypoints (the first 17 of the Goliath-308 set) → SMPL-X
# body joint indices. Eyes/ears (COCO 1-4) carry no SMPL-X joint.
COCO_TO_SMPLX = {
    0: 15,                       # nose      → head
    5: 16, 6: 17,                # shoulders
    7: 18, 8: 19,                # elbows
    9: 20, 10: 21,               # wrists
    11: 1, 12: 2,                # hips
    13: 4, 14: 5,                # knees
    15: 7, 16: 8,                # ankles
}


def fit_pose(
    kp: np.ndarray,
    sc: np.ndarray,
    *,
    gender: str,
    base_betas: np.ndarray,
    view: str,
    model_folder: str = "data/body_models",
    iters: int = 800,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Fit SMPL-X pose to one view's Sapiens keypoints.

    ``kp`` / ``sc`` are the 308-keypoint array and confidences.
    ``view`` ("front"|"side") only seeds ``global_orient``. Returns a
    dict with ``global_orient`` (3,), ``body_pose`` (63,), the camera
    ``scale`` / ``tu`` / ``tv``, and the reprojection RMS in pixels."""
    import smplx
    import torch

    dev = torch.device(device)
    kp = np.asarray(kp, np.float32)
    sc = np.asarray(sc, np.float32)

    idx_c = [c for c in COCO_TO_SMPLX]
    idx_s = [COCO_TO_SMPLX[c] for c in idx_c]
    tgt = torch.from_numpy(kp[idx_c]).to(dev)
    w = torch.from_numpy(sc[idx_c]).clamp(0, 1).to(dev)
    w[w < 0.3] = 0.0                       # ignore low-confidence joints

    num_betas = base_betas.shape[0]
    bm = smplx.create(model_path=model_folder, model_type="smplx",
                      gender=gender, num_betas=num_betas, use_pca=False,
                      flat_hand_mean=True, batch_size=1).to(dev)
    betas = torch.from_numpy(base_betas.astype(np.float32))[None].to(dev)

    go0 = [0.0, 0.0, 0.0] if view == "front" else [0.0, np.pi / 2, 0.0]
    global_orient = torch.tensor([go0], device=dev, requires_grad=True)
    body_pose = torch.zeros(1, 63, device=dev, requires_grad=True)
    scale = torch.tensor(300.0, device=dev, requires_grad=True)
    tu = torch.tensor(float(kp[idx_c, 0].mean()), device=dev,
                      requires_grad=True)
    tv = torch.tensor(float(kp[idx_c, 1].mean()), device=dev,
                      requires_grad=True)

    def joints():
        out = bm(betas=betas, body_pose=body_pose,
                 global_orient=global_orient,
                 transl=torch.zeros(1, 3, device=dev))
        return out.joints[0]

    def project(j):
        # image v axis points down → negate Y
        u = tu + scale * j[:, 0]
        v = tv - scale * j[:, 1]
        return torch.stack([u, v], -1)

    def reproj():
        p = project(joints()[idx_s])
        return p, ((p - tgt) ** 2).sum(-1)        # per-joint sq px error

    # Stage 1 — camera only (scale + translation), pose frozen.
    opt = torch.optim.Adam([scale, tu, tv], lr=2.0)
    for _ in range(150):
        opt.zero_grad()
        _, e = reproj()
        (w * e).sum().backward()
        opt.step()

    # Stage 2 — joint pose + camera.
    opt = torch.optim.Adam(
        [{"params": [body_pose, global_orient], "lr": 0.02},
         {"params": [scale, tu, tv], "lr": 1.0}])
    for i in range(iters):
        opt.zero_grad()
        _, e = reproj()
        data = (w * e).sum() / w.sum()
        prior = (body_pose ** 2).mean()
        (data + 60.0 * prior).backward()
        opt.step()
        if verbose and (i % 150 == 0 or i == iters - 1):
            rms = float(torch.sqrt((w * e).sum() / w.sum()))
            print(f"  pose-fit {view}  it {i:3d}  reproj RMS {rms:.1f}px")

    with torch.no_grad():
        _, e = reproj()
        rms = float(torch.sqrt((w * e).sum() / w.sum()))
    return {
        "global_orient": global_orient.detach().cpu().numpy()[0],
        "body_pose": body_pose.detach().cpu().numpy()[0],
        "scale": float(scale), "tu": float(tu), "tv": float(tv),
        "reproj_rms_px": rms,
    }
