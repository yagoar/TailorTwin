"""Differentiable soft-silhouette fit — match SMPL-X to two photos.

This is the render-based successor to ``silhouette_betas``. That module
fits SMPL-X betas to per-Y *width* and *depth* extents — two scalars per
slice. Two scalars are a bounding box: they pin how wide and how deep the
torso is, but throw away *where the bulk sits*. A belly that protrudes
forward and a flat back read the same ``depth`` as a centred, even
section, so the abdomen betas never get excited and the fitted waist
circumference falls toward the population mean (the observed −5 cm waist
/ highhip under-read).

Here the whole outline is matched instead. The SMPL-X mesh is splatted
into the front and side image grids as a soft occupancy mask and
compared, pixel for pixel, against the photo silhouettes. The side
view's *asymmetric* front/back boundary now drives the fit directly, so
the model is pushed onto the betas that actually reproduce the subject's
cross-section — circumference then reads honestly off the fitted 3-D
mesh, with the CAESAR shape prior filling only what the two silhouettes
genuinely leave free.

Girth zone, not whole body
--------------------------
Only the crotch→armpit band is fitted. The legs swing with the stance
and the head/shoulder seam is where arm removal is messiest; the torso
girth zone is near-rigid between a standing photo and the canonical
A-pose, so its silhouette is pure shape. It also holds every girth that
matters — bust, waist, highhip, hip. The projection scale is pinned to
the known metric scale (height + band cm) so the body cannot shrink
while the camera zooms to compensate.

The side view contributes only its depth profile (the per-row depth(Y)
curve): a 90° side photo has the near arm occluding the hip, so the raw
side silhouette is unreliable for pixel registration.

Differentiable rasteriser
-------------------------
No triangle rasteriser / no pytorch3d. Each torso triangle is sampled at
a fixed barycentric grid (the ~1k band vertices alone are too sparse to
fill a soft mask); the samples are bilinearly splatted into the grid and
a fixed Gaussian blur fills the gaps into a solid soft mask. Gradients
flow through the barycentric samples and the bilinear sub-pixel weights
to the betas.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

CANVAS_H = 256
CANVAS_W = 220
BODY_FRAC = 0.86          # torso-band height as a fraction of the canvas
ARM_CLASSES = (6, 15)     # Sapiens part-seg arm + hand class ids


# ----------------------------------------------------------------------
# target silhouettes — torso girth zone (crotch → armpit)
# ----------------------------------------------------------------------
def _largest_cc(m: np.ndarray) -> np.ndarray:
    import cv2
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n < 2:
        return m
    return lbl == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))


def _body_mask(seg_path: str, *, drop_arms: bool) -> np.ndarray:
    """Boolean body silhouette from a Sapiens part-seg map.

    ``drop_arms`` removes the arm + hand classes — correct for the front
    view, where the A-pose arms are clearly separated from the torso. It
    is WRONG for the side view: the near arm hangs over the torso, so
    deleting it splits the body into disconnected fragments and the
    largest-component filter then keeps only one. The side view keeps
    the whole silhouette (the down arm sits inside the torso depth
    envelope) and repairs the hand jut with the per-row convex fill."""
    seg = np.load(seg_path)
    if seg.ndim != 2:
        raise SystemExit(f"{seg_path}: expected a 2-D seg map")
    m = seg > 0
    if drop_arms:
        for c in ARM_CLASSES:
            m &= seg != c
    return _largest_cc(m)


def _row_runs(row: np.ndarray) -> list[tuple[int, int]]:
    cols = np.where(row)[0]
    if cols.size == 0:
        return []
    splits = np.where(np.diff(cols) > 1)[0]
    return [(int(b[0]), int(b[-1])) for b in np.split(cols, splits + 1)]


def detect_crotch(front_seg: str) -> float:
    """Crotch height in the front view, as a feet-fraction (0 = feet,
    1 = crown). The highest row where the body splits into two legs."""
    m = _body_mask(front_seg, drop_arms=True)
    ys, _ = np.where(m)
    y0, y1 = int(ys.min()), int(ys.max())
    hb = y1 - y0
    crotch = y0 + int(0.55 * hb)
    for r in range(y0 + int(0.40 * hb), y1):
        runs = _row_runs(m[r])
        if len(runs) >= 2 and (runs[-1][0] - runs[0][1]) > 0.015 * m.shape[1]:
            crotch = r
            break
    return (y1 - crotch) / hb


def build_target(
    seg_path: str, *, view: str, band: tuple[float, float],
    height_cm: float,
) -> tuple[np.ndarray, float]:
    """Photo torso-girth-zone silhouette → canvas mask + height in cm.

    ``band`` is ``(crotch_frac, armpit_frac)`` in feet-fractions — the
    girth zone holding bust, waist and hip, excluding the head, the legs
    and the shoulder/armpit seam.

    Both views drop the arm + hand classes. The front A-pose arms are
    cleanly separated, so removal just leaves the torso. The side near
    arm hangs *over* the torso at mid-depth: removing it punches a hole
    but does not change the front/back extent — so each band row is
    convex-filled left↔right, which both restores the true torso depth
    and reconnects the fragments the hole left behind. The component
    filter then runs *after* that repair."""
    import cv2

    seg = np.load(seg_path)
    if seg.ndim != 2:
        raise SystemExit(f"{seg_path}: expected a 2-D seg map")
    body = _largest_cc(seg > 0)              # full body — for Y extent
    ys, _ = np.where(body)
    y0, y1 = int(ys.min()), int(ys.max())
    hb = y1 - y0
    lo, hi = band
    r_bot = int(round(y1 - lo * hb))
    r_top = int(round(y1 - hi * hb))
    band_cm = height_cm * (hi - lo)

    m = seg > 0                              # arm-free, NOT yet filtered
    for c in ARM_CLASSES:
        m &= seg != c
    band_m = np.zeros_like(m)
    band_m[r_top:r_bot + 1] = m[r_top:r_bot + 1]
    if view == "side":                       # repair near-arm hole
        for r in range(r_top, r_bot + 1):
            cols = np.where(band_m[r])[0]
            if cols.size >= 2:
                band_m[r, cols[0]:cols[-1] + 1] = True
    band_m = _largest_cc(band_m)

    ys, xs = np.where(band_m)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = band_m[y0:y1 + 1, x0:x1 + 1].astype(np.uint8)
    bh, bw = crop.shape
    band_px = BODY_FRAC * CANVAS_H
    s = band_px / bh
    rw, rh = max(int(round(bw * s)), 1), int(round(band_px))
    rw = min(rw, CANVAS_W - 2)
    res = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((CANVAS_H, CANVAS_W), np.float32)
    oy = (CANVAS_H - rh) // 2
    ox = (CANVAS_W - rw) // 2
    canvas[oy:oy + rh, ox:ox + rw] = res
    return canvas, float(band_cm)


# ----------------------------------------------------------------------
# differentiable soft rasteriser
# ----------------------------------------------------------------------
def _gauss_kernel(sigma: float, device):
    import torch
    r = max(int(round(3 * sigma)), 1)
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def soft_silhouette(uv, H, W, kernel, *, occ_k=8.0):
    """Bilinearly splat projected vertices ``uv`` (N,2 pixels) → soft mask.

    Differentiable in ``uv``: the four-corner bilinear weights carry the
    sub-pixel gradient. The Gaussian blur fills inter-vertex gaps into a
    solid mask; ``1 - exp(-k·acc)`` squashes the splat accumulation into
    a [0,1] occupancy."""
    import torch
    import torch.nn.functional as F

    u, v = uv[:, 0], uv[:, 1]
    u0 = torch.floor(u)
    v0 = torch.floor(v)
    fu, fv = u - u0, v - v0
    u0 = u0.long().clamp(0, W - 2)
    v0 = v0.long().clamp(0, H - 2)
    acc = torch.zeros(H * W, dtype=uv.dtype, device=uv.device)
    for du in (0, 1):
        for dv in (0, 1):
            wu = fu if du else (1.0 - fu)
            wv = fv if dv else (1.0 - fv)
            idx = (v0 + dv) * W + (u0 + du)
            acc = acc.index_add(0, idx, wu * wv)
    acc = acc.view(1, 1, H, W)
    pad = kernel.shape[-1] // 2
    blur = F.conv2d(acc, kernel, padding=pad)
    return 1.0 - torch.exp(-occ_k * blur[0, 0])


# ----------------------------------------------------------------------
# fit
# ----------------------------------------------------------------------
def _a_pose(shoulder_deg: float) -> np.ndarray:
    from .refine_to_tape import _build_a_pose
    return _build_a_pose(shoulder_deg).astype(np.float32)


def fit_silhouette_render(
    base_betas: np.ndarray,
    gender: str,
    front_seg: str,
    side_seg: str,
    *,
    height_cm: float,
    model_folder: str = "data/body_models",
    a_pose_shoulder_deg: float = 30.0,
    n_active: int = 45,
    iters: int = 800,
    device: str = "cpu",
    verbose: bool = True,
    debug_dir: str | None = None,
) -> dict:
    """Fit SMPL-X betas so the mesh silhouette matches both photos.

    Returns a dict with ``betas`` (full vector), ``verts`` (A-pose mesh,
    scaled to ``height_cm``), ``joints`` and the per-view camera."""
    import smplx
    import torch

    dev = torch.device(device)
    crotch_frac = detect_crotch(front_seg)
    armpit_frac = min(crotch_frac + 0.30, 0.78)
    band = (crotch_frac, armpit_frac)
    tgt_f, band_cm = build_target(front_seg, view="front", band=band,
                                  height_cm=height_cm)
    tgt_s, _ = build_target(side_seg, view="side", band=band,
                            height_cm=height_cm)
    Tf = torch.from_numpy(tgt_f).to(dev)
    Ts = torch.from_numpy(tgt_s).to(dev)
    if verbose:
        print(f"  girth zone: crotch {crotch_frac:.2f} → armpit "
              f"{armpit_frac:.2f} of height  ({band_cm:.1f} cm)")

    num_betas = base_betas.shape[0]
    n_active = min(n_active, num_betas)
    bm = smplx.create(model_path=model_folder, model_type="smplx",
                      gender=gender, num_betas=num_betas, use_pca=False,
                      flat_hand_mean=True, batch_size=1).to(dev)
    body_pose = torch.from_numpy(
        _a_pose(a_pose_shoulder_deg).reshape(1, -1)).to(dev)

    # Arms are dropped (photo silhouettes are arm-free); the girth-zone
    # Y-band then keeps only the torso between crotch and armpit.
    from ..measure.regions import region_vertex_mask
    no_arm = ~region_vertex_mask(("left_arm", "right_arm"),
                                 model_folder=model_folder, gender=gender)

    base_b = torch.from_numpy(base_betas.astype(np.float32)).to(dev)
    d_betas = torch.zeros(n_active, device=dev, requires_grad=True)
    body_px = BODY_FRAC * CANVAS_H
    kernel = _gauss_kernel(3.0, dev)

    def forward_verts():
        out = bm(betas=torch.cat([base_b[:n_active] + d_betas,
                                  base_b[n_active:]])[None],
                 body_pose=body_pose, global_orient=torch.zeros(1, 3, device=dev),
                 transl=torch.zeros(1, 3, device=dev))
        return out.vertices[0], out.joints[0]

    # Dense surface sampling. The torso Y-band holds only ~1k vertices —
    # too sparse for a soft mask to fill (the render reads as dots). So
    # each torso triangle is sampled at a fixed barycentric grid; the
    # samples are linear in the vertices, hence differentiable in betas.
    faces = bm.faces.astype(np.int64)
    with torch.no_grad():
        v0 = forward_verts()[0].cpu().numpy()
    y0 = v0[:, 1]
    ymin0, h0 = y0.min(), y0.max() - y0.min()
    fcent = y0[faces].mean(1)
    f_ok = (no_arm[faces].all(1)
            & (fcent >= ymin0 + (crotch_frac - 0.06) * h0)
            & (fcent <= ymin0 + (armpit_frac + 0.06) * h0))
    torso_faces = torch.from_numpy(faces[f_ok]).to(dev)
    bary = []
    nb = 3
    for i in range(nb + 1):
        for j in range(nb + 1 - i):
            bary.append((i / nb, j / nb, 1 - i / nb - j / nb))
    bary = torch.tensor(bary, dtype=torch.float32, device=dev)  # (B,3)

    def band_verts(verts):
        """Dense points on the torso faces, trimmed to the crotch→armpit
        Y-band of the current mesh."""
        tri = verts[torso_faces]                       # (F,3,3)
        pts = torch.einsum("bk,fkc->fbc", bary, tri).reshape(-1, 3)
        y = pts[:, 1]
        ymin = verts[:, 1].min().detach()
        h = verts[:, 1].max().detach() - ymin
        sel = (y >= ymin + crotch_frac * h) & (y <= ymin + armpit_frac * h)
        return pts[sel]

    # Projection scale is FIXED to the known metric scale: the target
    # band fills ``body_px`` pixels and spans ``band_cm`` — so px-per-m
    # is body_px / (band_cm/100). A free scale would let the mesh shrink
    # while the camera zooms in to compensate, collapsing the fit. With
    # the scale pinned, the render directly penalises a wrong-sized body.
    scale_px_m = body_px / (band_cm / 100.0)

    def project(verts, c):
        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        cy, sy = torch.cos(c["yaw"]), torch.sin(c["yaw"])
        xr = x * cy + z * sy
        u = c["cu"] + scale_px_m * xr
        v = c["cv"] - scale_px_m * y
        return torch.stack([u, v], -1)

    # Camera init — only translation + yaw are free; centre the base
    # mesh's projected band on the canvas (the target band is centred).
    cam = {}
    with torch.no_grad():
        vb0 = band_verts(forward_verts()[0])
        for name, yaw0 in (("front", 0.0), ("side", np.pi / 2)):
            cy, sy = np.cos(yaw0), np.sin(yaw0)
            xr = (vb0[:, 0] * cy + vb0[:, 2] * sy).cpu().numpy()
            yy = vb0[:, 1].cpu().numpy()
            xc = 0.5 * (xr.max() + xr.min())
            yc = 0.5 * (yy.max() + yy.min())
            cam[name] = {
                "cu": torch.tensor(CANVAS_W / 2 - scale_px_m * xc,
                                   device=dev, requires_grad=True),
                "cv": torch.tensor(CANVAS_H / 2 + scale_px_m * yc,
                                   device=dev, requires_grad=True),
                "yaw": torch.tensor(float(yaw0), device=dev,
                                    requires_grad=True),
            }

    # Per-row foreground count = body width (front) / depth (side) at
    # each height. Integrating a whole scan-line averages out the soft
    # edge band, so this term carries the clean shape gradient that bare
    # pixel-MSE buries; pixel-MSE then only has to keep x/y registered.
    Tf_row = Tf.sum(1)
    Ts_row = Ts.sum(1)

    def render_loss():
        verts, _ = forward_verts()
        vb = band_verts(verts)
        rf = soft_silhouette(project(vb, cam["front"]),
                             CANVAS_H, CANVAS_W, kernel)
        rs = soft_silhouette(project(vb, cam["side"]),
                             CANVAS_H, CANVAS_W, kernel)
        lf = ((rf - Tf) ** 2).mean()
        ls = ((rs - Ts) ** 2).mean()
        rwf = (((rf.sum(1) - Tf_row) / CANVAS_W) ** 2).mean()
        rws = (((rs.sum(1) - Ts_row) / CANVAS_W) ** 2).mean()
        return lf, ls, rwf, rws, verts

    cam_params = [p for c in cam.values() for p in c.values()]

    def fit_camera(steps: int):
        o = torch.optim.Adam(cam_params, lr=0.05)
        for _ in range(steps):
            o.zero_grad()
            lf, ls, _, _, _ = render_loss()
            (lf + ls).backward()
            o.step()

    # Stage 1 — camera only. Resolve the side-view turn direction first
    # (subject's left vs right), then align translation + yaw.
    fit_camera(60)
    with torch.no_grad():
        best = None
        for yaw in (np.pi / 2, -np.pi / 2):
            cam["side"]["yaw"].copy_(torch.tensor(float(yaw), device=dev))
            _, ls, _, _, _ = render_loss()
            if best is None or ls.item() < best[0]:
                best = (ls.item(), yaw)
        cam["side"]["yaw"].copy_(torch.tensor(float(best[1]), device=dev))
    fit_camera(200)
    if verbose:
        lf, ls, _, _, _ = render_loss()
        print(f"  stage1 (camera): front {lf.item():.4f}  "
              f"side {ls.item():.4f}  side-yaw {best[1]:+.2f}")

    # Stage 2 — betas only, camera frozen. The torso is near-rigid, so
    # the stage-1 camera is trustworthy; freezing it stops the camera
    # absorbing shape error. Best iterate is kept (the soft-mask loss is
    # mildly non-convex and can drift late).
    #
    # The side view contributes only its depth profile (the per-row
    # term): the near arm occludes the hip in a 90° side photo, so the
    # raw side silhouette is unreliable for x/y registration — but the
    # row-integrated depth(Y) curve survives the convex-fill repair.
    opt = torch.optim.Adam([d_betas], lr=0.012)
    best_loss, best_db = 1e9, d_betas.detach().clone()
    for i in range(iters):
        opt.zero_grad()
        lf, ls, rwf, rws, verts = render_loss()
        vb = band_verts(verts)
        band_h = (vb[:, 1].max() - vb[:, 1].min()) * 100.0
        l_h = ((band_h - band_cm) / 100.0) ** 2
        l_prior = (d_betas ** 2).mean()
        loss = (lf + 30.0 * rwf + 30.0 * rws
                + 0.5 * l_h + 0.002 * l_prior)
        loss.backward()
        opt.step()
        if loss.item() < best_loss:
            best_loss, best_db = loss.item(), d_betas.detach().clone()
        if verbose and (i % 100 == 0 or i == iters - 1):
            print(f"  it {i:4d}  front {lf.item():.4f}  side {ls.item():.4f}"
                  f"  rowF {rwf.item():.5f}  rowS {rws.item():.5f}  "
                  f"band {band_h.item():.1f}  |db| "
                  f"{d_betas.detach().abs().mean().item():.3f}")
    with torch.no_grad():
        d_betas.copy_(best_db)
    if verbose:
        print(f"  best loss {best_loss:.4f}")

    # Diagnostic overlays — fitted render (red) over the photo (green).
    if debug_dir is not None:
        import cv2
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            verts_d = forward_verts()[0]
            for nm, T in (("front", Tf), ("side", Ts)):
                occ = soft_silhouette(project(band_verts(verts_d), cam[nm]),
                                      CANVAS_H, CANVAS_W, kernel)
                im = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
                im[..., 1] = (T.cpu().numpy() * 255).astype(np.uint8)
                im[..., 2] = (occ.cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(f"{debug_dir}/cmp_{nm}.png", cv2.resize(
                    im, (CANVAS_W * 2, CANVAS_H * 2),
                    interpolation=cv2.INTER_NEAREST))

    with torch.no_grad():
        betas = base_b.clone()
        betas[:n_active] = base_b[:n_active] + d_betas
        out = bm(betas=betas[None], body_pose=body_pose,
                 global_orient=torch.zeros(1, 3, device=dev),
                 transl=torch.zeros(1, 3, device=dev))
        verts = out.vertices[0].cpu().numpy().astype(np.float64)
        joints = out.joints[0].cpu().numpy().astype(np.float32)

    # Final uniform Y-scale to the exact stature (a pure Y stretch; the
    # height term above keeps this factor within ~1 %).
    y_min = float(verts[:, 1].min())
    cur_h = (verts[:, 1].max() - y_min) * 100.0
    s_h = height_cm / cur_h
    verts[:, 1] = y_min + (verts[:, 1] - y_min) * s_h
    if verbose:
        print(f"  height: {cur_h:.1f} → ×{s_h:.4f} → {height_cm} cm")

    return {
        "betas": betas.cpu().numpy().astype(np.float32),
        "verts": verts.astype(np.float32),
        "joints": joints,
        "body_pose": _a_pose(a_pose_shoulder_deg),
        "side_yaw": float(cam["side"]["yaw"].item()),
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("front", type=Path, help="front Sapiens '*_seg.npy'")
    p.add_argument("side", type=Path, help="side (90°) Sapiens '*_seg.npy'")
    p.add_argument("--height", type=float, required=True)
    p.add_argument("--base-fit", type=Path, required=True)
    p.add_argument("--out-prefix", type=Path, required=True)
    p.add_argument("--model-folder", default="data/body_models")
    p.add_argument("--n-active", type=int, default=45)
    p.add_argument("--iters", type=int, default=800)
    p.add_argument("--a-pose-shoulder-deg", type=float, default=30.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--debug-dir", default=None,
                   help="write fitted-render vs photo overlays here")
    args = p.parse_args(argv)

    for ph in (args.front, args.side, args.base_fit):
        if not ph.exists():
            raise SystemExit(f"missing input: {ph}")

    fit = np.load(args.base_fit)
    from ..fit.fit import fit_gender
    gender = fit_gender(fit)
    base_betas = fit["betas"].astype(np.float64)
    num_betas = base_betas.shape[0]
    print(f"silhouette-render fit: gender={gender}, betas={num_betas}")

    res = fit_silhouette_render(
        base_betas, gender, str(args.front), str(args.side),
        height_cm=args.height, model_folder=args.model_folder,
        a_pose_shoulder_deg=args.a_pose_shoulder_deg,
        n_active=args.n_active, iters=args.iters, device=args.device,
        debug_dir=args.debug_dir)

    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_npz = out_prefix.with_name(out_prefix.name + "_smplx_fit.npz")
    payload = {k: fit[k] for k in fit.files}
    payload["betas"] = res["betas"]
    payload["smplx_vertices"] = res["verts"]
    payload["smplx_joints"] = res["joints"]
    payload["body_pose"] = res["body_pose"]
    payload["global_orient"] = np.zeros((3,), dtype=np.float32)
    payload["transl"] = np.zeros((3,), dtype=np.float32)
    payload["z"] = np.array([])
    np.savez(out_npz, **payload)
    print(f"wrote {out_npz}")

    out_csv = out_prefix.with_name(out_prefix.name + "_measurements.csv")
    out_obj = out_prefix.with_name(out_prefix.name + "_fit_body.obj")
    out_smis = out_prefix.with_name(out_prefix.name + ".smis")
    cmd = [sys.executable, "-m", "tailor_twin.measure.cli", str(out_npz),
           "--num-betas", str(num_betas), "--gender", gender,
           "--model-folder", args.model_folder,
           "--save-csv", str(out_csv), "--save-obj", str(out_obj),
           "--save-smis", str(out_smis)]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
