"""Fuse the front + side Sapiens pointmaps into one body point cloud.

The slice/betas fit collapses the pointmap to 24 width/depth numbers —
many bodies hit the same 24, so the fitted mesh need not resemble the
subject. The dense path keeps the **surface**: every pointmap pixel is
a metric 3-D point; a chamfer fit of SMPL-X to that cloud is pinned by
the whole body, not a handful of rings.

This module builds the fused cloud. Each view's pointmap is a partial
shell — front sees the front surface, side sees one lateral surface.
They are registered into a shared body frame:

* axis map + metric scale (body height → ``height_cm``), Y flipped up;
* the side shell is yaw-rotated ~90° into the front frame;
* ICP refines the side→front alignment on the overlap.

The result is a ~270° body cloud (back unseen) consumed by
``fit_scan``.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .pointmap import DOME29_ARM_CLASSES, _load_pointmap


def _refine_depth_with_normal(pm: np.ndarray, mask: np.ndarray,
                              normal: np.ndarray, *, anchor: float = 0.2,
                              verbose: bool = True) -> np.ndarray:
    """Re-integrate the pointmap depth (Z) from the Sapiens2 normal map.

    The pointmap X,Y are a near-perfect orthographic pixel grid; only Z
    (depth-from-camera) is network-guessed and noisy — and the body's
    forward bulge (belly, chest) lives entirely on that weak axis. The
    Sapiens2 normal map measures surface *orientation* directly and is
    far cleaner.

    For an orthographic depth map the normal implies the depth gradient
    ``Zc = -sx·nx/nz``, ``Zr = -sy·ny/nz``. We solve a screened-Poisson
    least-squares system so the refined Z's gradient matches that field
    while staying anchored (weight ``anchor``) to the original Z — which
    keeps global depth and metric scale intact. Returns ``pm`` with Z
    replaced inside ``mask``.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import lsqr

    h, w, _ = pm.shape
    sx = float(np.median(np.diff(pm[h // 2, :, 0])))
    sy = float(np.median(np.diff(pm[:, w // 2, 1])))

    nz = normal[..., 2].astype(np.float64).copy()
    small = np.abs(nz) < 0.2                       # grazing — avoid /0 blow-up
    nz[small] = 0.2 * np.where(nz[small] >= 0, 1.0, -1.0)
    gx = -sx * normal[..., 0] / nz                 # target dZ/dcol
    gy = -sy * normal[..., 1] / nz                 # target dZ/drow

    idx = np.full((h, w), -1, dtype=np.int64)
    ys, xs = np.where(mask)
    n = len(ys)
    idx[ys, xs] = np.arange(n)
    z0 = pm[..., 2]

    hx = mask[:, :-1] & mask[:, 1:]                # horizontal-neighbour pairs
    yi, xi = np.where(hx)
    vy = mask[:-1, :] & mask[1:, :]                # vertical-neighbour pairs
    yj, xj = np.where(vy)
    n_h, n_v = len(yi), len(yj)
    wa = float(np.sqrt(anchor))

    r_h = np.arange(n_h)
    r_v = n_h + np.arange(n_v)
    r_a = n_h + n_v + np.arange(n)
    rows = np.concatenate([r_h, r_h, r_v, r_v, r_a])
    cols = np.concatenate([idx[yi, xi + 1], idx[yi, xi],
                           idx[yj + 1, xj], idx[yj, xj], np.arange(n)])
    vals = np.concatenate([np.ones(n_h), -np.ones(n_h),
                           np.ones(n_v), -np.ones(n_v), np.full(n, wa)])
    b = np.concatenate([gx[yi, xi], gy[yj, xj], wa * z0[ys, xs]])
    A = sp.coo_matrix((vals, (rows, cols)),
                      shape=(n_h + n_v + n, n)).tocsr()
    z_new = lsqr(A, b, atol=1e-6, btol=1e-6, iter_lim=400)[0]

    out = pm.copy()
    out[ys, xs, 2] = z_new
    if verbose:
        shift = (z_new - z0[ys, xs])
        print(f"  normal depth refine: {n} px  Δdepth mean "
              f"{shift.mean() * 100:+.2f} cm  std {shift.std() * 100:.2f} cm")
    return out


def _view_cloud(ply: Path, seg_path: Path, height_cm: float,
                *, normal_npy: Path | None = None) -> np.ndarray:
    """One view's body points in a metric, Y-up camera frame.

    Pointmap camera axes are (x = image-horizontal, y = down,
    z = depth-from-camera). Returns (N, 3) with Y flipped up and the
    cloud scaled so the body's vertical span equals ``height_cm`` (m).
    Arms are kept — their scan points anchor the body's lateral extent,
    which a one-directional chamfer otherwise leaves unconstrained.

    If ``normal_npy`` is given, the depth axis is first re-integrated
    from that Sapiens2 normal map (see :func:`_refine_depth_with_normal`).
    """
    seg = np.load(seg_path)
    h, w = seg.shape
    pm = _load_pointmap(ply, h, w)
    body = seg > 0
    if normal_npy is not None:
        pm = _refine_depth_with_normal(pm, body, np.load(normal_npy))
    P = pm[body].astype(np.float64)
    P[:, 1] *= -1.0                                   # camera Y is down
    lo, hi = np.percentile(P[:, 1], [0.5, 99.5])
    P *= (height_cm / 100.0) / (hi - lo)              # metric, m
    P -= P.mean(axis=0)
    return P


def _yaw(P: np.ndarray, deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return P @ R.T


def fuse_body_cloud(
    front_ply: Path, front_seg: Path,
    side_ply: Path, side_seg: Path,
    height_cm: float,
    *, verbose: bool = True,
) -> np.ndarray:
    """Front + side pointmaps → one registered metric body cloud (m)."""
    import open3d as o3d

    front = _view_cloud(front_ply, front_seg, height_cm)
    side = _view_cloud(side_ply, side_seg, height_cm)

    def to_o3d(P):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(P)
        return pc

    tgt = to_o3d(front)
    # The side shell is ~90° off the front. Try both yaw signs, keep the
    # one ICP locks better (the body-side overlap is thin but present).
    best = None
    for deg in (90.0, -90.0):
        src = to_o3d(_yaw(side, deg))
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, 0.05, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=80))
        if best is None or reg.fitness > best[0]:
            best = (reg.fitness, reg.inlier_rmse, deg, reg.transformation)
    fitness, rmse, deg, T = best
    if verbose:
        print(f"side→front ICP: yaw {deg:+.0f}°  fitness {fitness:.3f}  "
              f"rmse {rmse * 100:.1f} cm")

    src = to_o3d(_yaw(side, deg))
    src.transform(T)
    fused = np.vstack([front, np.asarray(src.points)])
    # Y-up, feet at 0 — fit_scan crops by Y fraction.
    fused[:, 1] -= fused[:, 1].min()
    if verbose:
        ext = fused.max(0) - fused.min(0)
        print(f"fused cloud: {len(fused)} pts  bbox(cm) "
              f"{np.round(ext * 100, 1)}")
    return fused.astype(np.float32)


def _side_cloud_handfilled(ply: Path, seg_path: Path, height_cm: float,
                           *, normal_npy: Path | None = None,
                           verbose: bool = True) -> np.ndarray:
    """Side-view body cloud with the hand-occluded torso band rebuilt.

    In the simultaneous front+side capture the arm is ~30° out, so in
    the side view the hand crosses the torso around hip height. Arm/hand
    pixels are dropped — an SMPL-X torso vertex must not snap onto a
    hand point — which leaves a Y-band where the torso surface is
    missing. Within the arm's height range, height bins starved of
    torso points are flagged occluded; their front/back depth is
    interpolated from the clean bands above and below and re-synthesised
    as points, so the chamfer still has a hip/highhip surface to fit.
    """
    seg = np.load(seg_path)
    h, w = seg.shape
    pm = _load_pointmap(ply, h, w)
    body = seg > 0
    if normal_npy is not None:
        pm = _refine_depth_with_normal(pm, body, np.load(normal_npy))
    arm = np.isin(seg, DOME29_ARM_CLASSES)
    torso = body & ~arm

    # Metric scale from the full body Y-span (matches _view_cloud).
    Pf = pm[body].astype(np.float64)
    Pf[:, 1] *= -1.0
    lo, hi = np.percentile(Pf[:, 1], [0.5, 99.5])
    s = (height_cm / 100.0) / (hi - lo)

    def _scaled(mask):
        Q = pm[mask].astype(np.float64)
        Q[:, 1] *= -1.0
        Q *= s
        return Q

    P = _scaled(torso)
    mean = P.mean(axis=0)
    P -= mean
    arm_y = _scaled(arm)[:, 1] - mean[1] if arm.any() else np.array([])

    # Bin torso points by height; a bin inside the arm's Y-range that is
    # starved of points (vs the median) is hand-occluded.
    y0, y1 = P[:, 1].min(), P[:, 1].max()
    nb = max(20, int((y1 - y0) / 0.01))
    edges = np.linspace(y0, y1, nb + 1)
    yc = 0.5 * (edges[:-1] + edges[1:])
    binc = np.clip(np.digitize(P[:, 1], edges) - 1, 0, nb - 1)
    xlo = np.full(nb, np.nan)
    xhi = np.full(nb, np.nan)
    zmid = np.full(nb, np.nan)
    cnt = np.zeros(nb, int)
    for b in range(nb):
        m = binc == b
        cnt[b] = int(m.sum())
        if cnt[b] >= 12:
            xlo[b] = np.percentile(P[m, 0], 2)
            xhi[b] = np.percentile(P[m, 0], 98)
            zmid[b] = P[m, 2].mean()
    # The hand does not starve the bin of points — the torso keeps a
    # front+back sliver around it — it collapses the torso *depth*
    # (front-to-back extent). Flag bins whose depth dips well below the
    # surrounding clean torso, inside the arm's height band.
    depth = xhi - xlo
    valid = ~np.isnan(depth)
    win = 12
    ref = np.array([
        np.nanmax(depth[max(0, b - win):b + win + 1])
        if valid[max(0, b - win):b + win + 1].any() else np.nan
        for b in range(nb)])
    occ = valid & (depth < 0.78 * ref)
    if arm_y.size:                       # only inside the arm's height band
        occ &= (yc >= arm_y.min()) & (yc <= arm_y.max())
    good = valid & (~occ)
    if verbose:
        print(f"  side hand-fill: {int(occ.sum())}/{nb} torso bins "
              f"occluded — interpolated")
    if good.sum() < 4 or not occ.any():
        return P.astype(np.float32)

    xlo_i = np.interp(yc, yc[good], xlo[good])
    xhi_i = np.interp(yc, yc[good], xhi[good])
    zmid_i = np.interp(yc, yc[good], zmid[good])
    synth = [np.stack([np.linspace(xlo_i[b], xhi_i[b], 24),
                       np.full(24, yc[b]), np.full(24, zmid_i[b])], axis=1)
             for b in np.where(occ)[0]]
    return np.vstack([P, *synth]).astype(np.float32)


def _poisson_surface(cloud: np.ndarray, *, depth: int = 9,
                     density_quantile: float = 0.05,
                     verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Watertight surface mesh from the fused 360° point cloud.

    The raw 4-shell cloud is a ~2-3 cm-thick noisy band (per-shell ICP
    rmse ~2 cm). Chamfering SMPL-X to it point-to-point lets the mesh
    drift to the band's *outer* envelope (inflation) or, one-directional,
    bulge straight through it. Poisson reconstruction collapses the band
    to a single smooth surface; the SMPL-X fit then uses a point-to-
    *surface* chamfer against it, which pulls a bulged belly back in and
    a flat seat back out. Low-density vertices — the balloons Poisson
    grows over the unseen head-crown / soles / armpits — are cropped.
    """
    import open3d as o3d

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(cloud.astype(np.float64))
    pc.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
    pc.orient_normals_consistent_tangent_plane(40)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, density_quantile))
    mesh.remove_unreferenced_vertices()
    v = np.asarray(mesh.vertices, np.float32)
    f = np.asarray(mesh.triangles, np.int64)
    if verbose:
        ext = v.max(0) - v.min(0)
        print(f"poisson surface: {len(v)} verts  {len(f)} faces  "
              f"bbox(cm) {np.round(ext * 100, 1)}")
    return v, f


def _register_shell_to_mesh(shell: np.ndarray, mesh_v: np.ndarray,
                            yaw_candidates: tuple[float, ...],
                            name: str) -> np.ndarray:
    """Yaw a partial shell into the front frame and ICP it to the mesh.

    The shell is registered to the *fitted full-body mesh* (not to
    another shell) — a complete body gives ICP a far stronger lock than
    thin shell-to-shell overlap. The shell is first Y-locked (its
    feet→crown stretched onto the mesh's) and ICP is confined to the
    horizontal plane, so a vertical slip cannot inflate stature. ICP is
    point-to-plane (slides along the surface tangent, not toward the
    nearest vertex) — it locks the shell depth tighter than
    point-to-point.

    A partial back/side shell overlaps the mesh on a thin band, so a
    single centroid-aligned ICP start drops into the wrong depth basin
    (this collapsed the 1b back shell ~4 cm inward). ICP is therefore
    run from a sweep of initial depth (Z) offsets — and yaw candidates —
    and the highest-fitness solution wins.
    """
    import open3d as o3d

    def _pc(P, *, normals: bool = False):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.asarray(P, np.float64))
        if normals:
            pc.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        return pc

    tgt = _pc(mesh_v, normals=True)            # point-to-plane needs them
    p2pl = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=200)
    m_lo, m_hi = mesh_v[:, 1].min(), mesh_v[:, 1].max()
    depth = float(mesh_v[:, 2].max() - mesh_v[:, 2].min())
    best = None
    for deg in yaw_candidates:
        s0 = _yaw(shell, deg)
        s_lo, s_hi = s0[:, 1].min(), s0[:, 1].max()
        s0[:, 1] = m_lo + (s0[:, 1] - s_lo) * (m_hi - m_lo) / (s_hi - s_lo)
        s0[:, [0, 2]] += mesh_v[:, [0, 2]].mean(0) - s0[:, [0, 2]].mean(0)
        for dz in (-0.6, -0.3, 0.0, 0.3, 0.6):     # depth-basin sweep
            si = s0.copy()
            si[:, 2] += dz * depth
            reg = o3d.pipelines.registration.registration_icp(
                _pc(si), tgt, 0.05, np.eye(4), p2pl, crit)
            if best is None or reg.fitness > best[0]:
                best = (reg.fitness, reg.inlier_rmse, deg, dz,
                        si, reg.transformation)
    fitness, rmse, deg, dz, s0, T = best
    print(f"{name}→mesh ICP: yaw {deg:+.0f}°  dz {dz:+.1f}  "
          f"fitness {fitness:.3f}  rmse {rmse * 100:.1f} cm")
    pc = _pc(s0)
    pc.transform(T)
    reg = np.asarray(pc.points)
    reg[:, 1] = np.clip(reg[:, 1], m_lo, m_hi)            # no Y overhang
    return reg.astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    """CLI: 360° dense chamfer fit of SMPL-X to a 4-photo capture.

    The four Sapiens pointmaps (front / back / left / right) are dense
    metric shells of the body surface. Pass 1 fits SMPL-X to the front
    shell for a full body mesh; the back/left/right shells are then
    registered to that mesh and pass 2 refits to the ~360° cloud — so
    every side (depth included) is *measured*, not prior-guessed.

    Input is a capture folder holding ``front.jpg``/``back.jpg``/
    ``left.jpg``/``right.jpg`` (the layout the capture webapp writes).
    """
    import argparse
    import json as _json
    import subprocess
    import sys as _sys

    p = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    p.add_argument("capture", type=Path,
                   help="Capture folder with front/back/left/right.jpg.")
    p.add_argument("--height", type=float, default=None,
                   help="Stature cm — else read from capture_meta.json.")
    p.add_argument("--base-fit", type=Path, required=True,
                   help="Existing fit npz — supplies gender + num_betas.")
    p.add_argument("--out-prefix", type=Path, required=True)
    p.add_argument("--model-folder", default="data/body_models")
    p.add_argument("--device", default="cpu")
    p.add_argument("--model-size", default="0.4b",
                   help="Pointmap model size — 1b gives a sharper depth "
                        "axis (the belly/chest accuracy).")
    p.add_argument("--seg-size", default=None,
                   help="Part-seg model size (default = --model-size).")
    p.add_argument("--normal-size", default="0.4b",
                   help="Normal model size for the depth refinement.")
    p.add_argument("--no-normals", dest="use_normals", action="store_false",
                   help="Skip the Sapiens2 normal-map depth refinement.")
    p.add_argument("--apose-shoulder-deg", type=float, default=55.0,
                   help="Arm drop of the canonical A-pose the measurements "
                        "are taken on (55° = a clear A).")
    p.set_defaults(use_normals=True)
    args = p.parse_args(argv)

    cap = args.capture
    # Accept any subset of views — a 4-shot 360° folder, or a 2-phone
    # paired front+side folder. ``front`` is required; the rest are
    # registered to the front-fitted mesh as they are present.
    all_views = ("front", "back", "left", "right", "side")
    photos = {v: cap / f"{v}.jpg" for v in all_views
              if (cap / f"{v}.jpg").exists()}
    views = tuple(photos)
    if "front" not in views:
        raise SystemExit(f"no front.jpg in {cap}")
    if not args.base_fit.exists():
        raise SystemExit(f"missing input: {args.base_fit}")
    print(f"views: {', '.join(views)}")

    height = args.height
    meta_path = cap / "capture_meta.json"
    if height is None and meta_path.exists():
        meta = _json.loads(meta_path.read_text())
        height = float(meta.get("height_cm") or 0) or None
    if height is None:
        raise SystemExit("no --height and no height_cm in capture_meta.json")
    print(f"capture {cap.name}  height {height} cm")

    work_dir = args.out_prefix.with_name(args.out_prefix.name + "_sapiens")
    from .pointmap import run_sapiens
    art = run_sapiens(photos, work_dir, model_size=args.model_size,
                      seg_size=args.seg_size, device=args.device)

    # Sapiens2 normal maps → re-integrate the noisy pointmap depth axis.
    nrm: dict[str, Path | None] = {v: None for v in views}
    if args.use_normals:
        from .pointmap import run_normal
        wd = Path(work_dir).resolve()
        nart = run_normal(wd / "in", wd / "normal", wd / "out",
                          model_size=args.normal_size, device=args.device)
        nrm = {v: nart[v]["npy"] for v in views}
        print("normal-map depth refinement: ON")

    print("building front pointmap shell …")
    front = _view_cloud(art["front"]["ply"], art["front"]["seg"], height,
                        normal_npy=nrm["front"])
    front[:, 1] -= front[:, 1].min()
    front = front.astype(np.float32)
    print(f"front cloud: {len(front)} pts  "
          f"bbox(cm) {np.round((front.max(0) - front.min(0)) * 100, 1)}")

    fit = np.load(args.base_fit)
    from .fit import FitConfig, fit_scan, fit_gender, save_fit
    gender = fit_gender(fit)
    num_betas = int(fit["betas"].shape[0])
    cfg = FitConfig(model_folder=args.model_folder, gender=gender,
                    num_betas=num_betas, device=args.device,
                    partial_cloud=True, use_displacement=False)

    # Pass 1 — fit to the front shell. Gives a complete body mesh.
    print("\n=== pass 1: front-shell chamfer fit ===")
    result = fit_scan(front, cfg, verbose=True)

    # Register every non-front shell to the *fitted mesh* and refit to
    # the combined cloud — depth (and any back) become measured.
    others = [v for v in views if v != "front"]
    extra = [front]
    if others:
        print(f"\n=== registering {'/'.join(others)} shells "
              f"to the fitted mesh ===")
        mesh_v = result.smplx_vertices.astype(np.float64)
        # back = 180° turn; side/left/right = ±90° (ICP picks the sign).
        yaw_cand = {"back": (180.0,), "left": (90.0, -90.0),
                    "right": (90.0, -90.0), "side": (90.0, -90.0)}
        for v in others:
            if v == "side":
                shell = _side_cloud_handfilled(
                    art[v]["ply"], art[v]["seg"], height,
                    normal_npy=nrm[v])
            else:
                shell = _view_cloud(art[v]["ply"], art[v]["seg"], height,
                                    normal_npy=nrm[v])
            extra.append(
                _register_shell_to_mesh(shell, mesh_v, yaw_cand[v], v))
    combined = np.vstack(extra)
    print(f"\n=== pass 2: 360° chamfer fit ({len(combined)} pts) ===")
    # One-directional scan→mesh chamfer. A bidirectional point-to-point
    # variant and a Poisson point-to-surface variant were both tried:
    # the 4-shell cloud is a ~2-3 cm noisy band, so both inflated or
    # shrank the girths (and the surface variant distorted the head).
    # One-directional keeps the measurements tightest (~1.3 cm mean).
    result = fit_scan(combined, cfg, verbose=True)

    # The one-directional chamfer does not bound the mesh's overall size
    # — rescale uniformly so stature equals the measured height; girths
    # scale with it.
    verts = result.smplx_vertices.astype(np.float64)
    y_ext = float(verts[:, 1].max() - verts[:, 1].min())
    s = (height / 100.0) / y_ext
    result.smplx_vertices = (verts * s).astype(np.float32)
    result.smplx_joints = (result.smplx_joints * s).astype(np.float32)
    print(f"height normalise: {y_ext * 100:.1f} → ×{s:.4f} → "
          f"{height} cm")

    # Fitted-pose npz. The subject posed in an A-pose for the capture,
    # so this fitted mesh *is* an A-pose body — and the measure
    # pipeline's landmark detection is calibrated for fit-pipeline
    # poses. Body measurements are taken on this mesh; a mesh
    # synthetically reposed to a canonical A-pose breaks several measure
    # landmarks (verified — H28 → garbage, lowbust off ~9 cm).
    out_npz = args.out_prefix.with_name(
        args.out_prefix.name + "_smplx_fit.npz")
    save_fit(result, out_npz)
    print(f"wrote {out_npz}")

    # Chamfer-fit heatmap — fitted mesh vs the 360° scan cloud.
    try:
        from .render_overlays import chamfer_heatmap
        chamfer_heatmap(out_npz, Path(work_dir).resolve() / "out", height,
                        args.out_prefix.with_name(
                            args.out_prefix.name + "_chamfer_heatmap.png"),
                        model_folder=args.model_folder)
    except Exception as e:  # noqa: BLE001
        print(f"chamfer heatmap skipped: {e}")

    # Canonical A-pose OBJ — a visualisation deliverable only (same
    # betas, pose-normalised). NOT measured: see the note above.
    import smplx
    import torch

    from ..measure.exports import write_obj
    from .refine_to_tape import _build_a_pose
    bm = smplx.create(model_path=args.model_folder, model_type="smplx",
                      gender=gender, num_betas=num_betas, use_pca=False,
                      flat_hand_mean=True, batch_size=1)
    ap = _build_a_pose(args.apose_shoulder_deg).astype(np.float32)
    with torch.no_grad():
        o = bm(betas=torch.from_numpy(result.betas[None].astype(np.float32)),
               body_pose=torch.from_numpy(ap.reshape(1, 63)))
    av = o.vertices[0].numpy().astype(np.float64)
    av *= (height / 100.0) / float(av[:, 1].max() - av[:, 1].min())
    av[:, 1] -= av[:, 1].min()
    out_apose = args.out_prefix.with_name(
        args.out_prefix.name + "_apose.obj")
    write_obj(av, bm.faces.astype(np.int64), out_apose)
    print(f"wrote {out_apose}  (A-pose {args.apose_shoulder_deg:.0f}°, "
          f"visual only)")

    # Measurements + the deliverable OBJ — on the fitted A-pose mesh.
    out_csv = args.out_prefix.with_name(
        args.out_prefix.name + "_measurements.csv")
    out_obj = args.out_prefix.with_name(
        args.out_prefix.name + "_fit_body.obj")
    cmd = [_sys.executable, "-m", "tailor_twin.measure.cli", str(out_npz),
           "--num-betas", str(num_betas), "--gender", gender,
           "--model-folder", args.model_folder,
           "--save-csv", str(out_csv), "--save-obj", str(out_obj)]
    return subprocess.run(cmd).returncode


def _render(cloud: np.ndarray, out_png: str) -> None:
    """Quick 3-view scatter of the fused cloud for eyeballing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = cloud - cloud.mean(0)
    fig, ax = plt.subplots(1, 3, figsize=(13, 7))
    for a, (i, j, t) in zip(ax, [(0, 1, "front XY"), (2, 1, "side ZY"),
                                 (0, 2, "top XZ")]):
        a.scatter(c[:, i], c[:, j], s=1, c=c[:, 3 - i - j], cmap="viridis")
        a.set_aspect("equal"); a.set_title(t); a.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=100)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
