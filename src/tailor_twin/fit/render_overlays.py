"""Repose the fitted body to a canonical A-pose, and overlay the SMPL-X
model on each capture photo in that photo's *original* pose.

Two deliverables:

* :func:`repose_apose` — rebuild the fitted body from its betas in a
  canonical A-pose (shoulders 30° down). Betas are untouched, so every
  proportion is preserved; the mesh is uniformly rescaled to the
  measured stature. Output is one clean, pose-normalised OBJ.

* :func:`overlay_view` — the 4-photo capture holds the subject in four
  different poses (front/back arms out, sides arms down). A single
  fitted mesh cannot match all four, so each view is re-fitted to its
  own Sapiens2 pointmap shell — that shell *is* the Sapiens2 pose
  information for that frame. The re-posed mesh is then projected back
  onto the photo through the pointmap's pixel↔3-D correspondence and
  drawn as a translucent overlay.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .pointmap import _load_pointmap

VIEWS = ("front", "back", "left", "right")


def _view_cloud_px(ply: Path, seg_path: Path, height_cm: float):
    """One view's metric body cloud + the image pixel of every point.

    Mirrors :func:`pointmap_dense._view_cloud` (camera Y flipped up,
    scaled so the body's vertical span = ``height_cm``, mean-centred)
    but also returns the (row, col) pixel each cloud point came from —
    that is the pointmap's pixel↔3-D map, used to project a fitted mesh
    back onto the photo.
    """
    seg = np.load(seg_path)
    h, w = seg.shape
    pm = _load_pointmap(ply, h, w)
    body = seg > 0
    rows, cols = np.where(body)
    P = pm[body].astype(np.float64)
    P[:, 1] *= -1.0
    lo, hi = np.percentile(P[:, 1], [0.5, 99.5])
    P *= (height_cm / 100.0) / (hi - lo)
    P -= P.mean(axis=0)
    return P.astype(np.float32), rows, cols, h, w


def repose_apose(fit_npz: Path, out_obj: Path, height_cm: float,
                 *, shoulder_deg: float = 55.0,
                 model_folder: str = "data/body_models") -> Path:
    """Rebuild the fitted body in a canonical A-pose and write an OBJ.

    Proportions come straight from the fit's betas; only the pose is
    normalised. The mesh is uniformly scaled so its stature equals
    ``height_cm`` and the feet sit at Y=0.
    """
    import smplx
    import torch

    from ..measure.exports import write_obj
    from .refine_to_tape import _build_a_pose

    d = np.load(fit_npz)
    betas = d["betas"].astype(np.float32)
    gender = str(d["gender"])
    num_betas = int(betas.shape[0])

    bm = smplx.create(model_path=model_folder, model_type="smplx",
                       gender=gender, num_betas=num_betas, use_pca=False,
                       flat_hand_mean=True, batch_size=1)
    a_pose = _build_a_pose(shoulder_deg).astype(np.float32).reshape(1, 63)
    with torch.no_grad():
        out = bm(betas=torch.from_numpy(betas).unsqueeze(0),
                 body_pose=torch.from_numpy(a_pose))
    verts = out.vertices[0].numpy().astype(np.float64)
    faces = bm.faces.astype(np.int64)

    # Uniform rescale → stature == height_cm; feet to Y=0.
    y_ext = float(verts[:, 1].max() - verts[:, 1].min())
    verts *= (height_cm / 100.0) / y_ext
    verts[:, 1] -= verts[:, 1].min()
    write_obj(verts, faces, out_obj)
    print(f"A-pose obj: {out_obj}  stature {height_cm} cm  "
          f"({len(verts)} v, shoulders {shoulder_deg:.0f}°)")
    return out_obj


def overlay_view(view: str, ply: Path, seg_path: Path, photo: Path,
                 height_cm: float, cfg, faces: np.ndarray,
                 out_png: Path) -> Path:
    """Fit SMPL-X to one view's pointmap shell and overlay it on the photo.

    The per-view fit recovers that frame's actual pose from the Sapiens2
    pointmap. The mesh — in the shell's metric frame — is projected to
    image pixels by nearest-neighbour through the pointmap pixel↔3-D
    map, then drawn as a translucent filled silhouette: the
    camera-facing triangles are filled, giving uniform body coverage
    (a vertex scatter clumps on the dense hand/face verts and vanishes
    on the sparse torso).
    """
    from PIL import Image
    from scipy.spatial import cKDTree

    from .fit import fit_scan

    cloud, rows, cols, h, w = _view_cloud_px(ply, seg_path, height_cm)
    print(f"  [{view}] cloud {len(cloud)} pts → fitting pose …")
    result = fit_scan(cloud, cfg, verbose=False)
    verts = result.smplx_vertices.astype(np.float64)

    # Project each mesh vertex to the pixel of its nearest cloud point.
    _, idx = cKDTree(cloud).query(verts)
    px = np.column_stack([cols[idx], rows[idx]]).astype(np.float64)

    # Camera-facing cull: the pointmap camera looks along +Z, SMPL-X
    # faces are wound outward — keep triangles whose outward normal has
    # a negative Z (points back at the camera).
    tri = verts[faces]
    nz = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 2]
    front = faces[nz < 0]

    img = Image.open(photo).convert("RGB").resize((w, h))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(np.asarray(img))
    polys = px[front]                                   # (F, 3, 2)
    ax.add_collection(PolyCollection(
        polys, facecolors="#39ff14", edgecolors="none", alpha=0.32))
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_title(f"{view} — SMPL-X over capture (original pose)",
                 fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{view}] wrote {out_png}")
    return out_png


def chamfer_heatmap(fit_npz: Path, sapiens_dir: Path, height_cm: float,
                    out_png: Path, *,
                    model_folder: str = "data/body_models") -> Path:
    """Per-vertex chamfer-fit error of the final mesh, drawn as a heatmap.

    The four Sapiens2 pointmap shells are re-registered to the fitted
    mesh (yaw + Y-lock + XZ ICP, as the dense fit does) to rebuild the
    360° scan cloud, then every SMPL-X vertex is coloured by its
    distance to the nearest scan point. Four orthographic views; back-
    facing triangles culled per view so the surface error reads clean.
    """
    import smplx
    from scipy.spatial import cKDTree

    from .pointmap_dense import _register_shell_to_mesh, _view_cloud

    d = np.load(fit_npz)
    verts = d["smplx_vertices"].astype(np.float64)
    gender = str(d["gender"])
    faces = smplx.create(model_path=model_folder, model_type="smplx",
                         gender=gender, num_betas=10, use_pca=False,
                         batch_size=1).faces.astype(np.int64)

    # Rebuild the scan cloud in the fitted mesh's frame — using whatever
    # views the capture has (4-shot 360° or a 2-phone front+side pair).
    yaw_cand = {"front": (0.0,), "back": (180.0,),
                "left": (90.0, -90.0), "right": (90.0, -90.0),
                "side": (90.0, -90.0)}
    clouds = []
    for v in ("front", "back", "left", "right", "side"):
        if not (sapiens_dir / f"{v}.ply").exists():
            continue
        shell = _view_cloud(sapiens_dir / f"{v}.ply",
                            sapiens_dir / f"{v}_seg.npy", height_cm)
        clouds.append(_register_shell_to_mesh(shell, verts, yaw_cand[v], v))
    cloud = np.vstack(clouds)

    dist_cm = cKDTree(cloud).query(verts)[0] * 100.0
    print(f"chamfer error: mean {dist_cm.mean():.2f}  median "
          f"{np.median(dist_cm):.2f}  p95 {np.percentile(dist_cm, 95):.2f} cm")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    vmax = float(np.percentile(dist_cm, 95))
    face_d = dist_cm[faces].mean(axis=1)
    tri = verts[faces]
    # (projection axes, depth axis+sign, label) per view.
    panels = [((0, 1), (2, -1), "front"), ((0, 1), (2, +1), "back"),
              ((2, 1), (0, +1), "left"),  ((2, 1), (0, -1), "right")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 7))
    for ax, ((ax0, ax1), (dax, dsign), label) in zip(axes, panels):
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        front = (normal[:, dax] * dsign) < 0          # facing the camera
        polys = tri[front][:, :, [ax0, ax1]].copy()
        polys[:, :, 0] *= -1 if label in ("back", "right") else 1
        pc = PolyCollection(polys, array=face_d[front], cmap="turbo",
                            edgecolors="none")
        pc.set_clim(0, vmax)
        ax.add_collection(pc)
        ax.set_aspect("equal"); ax.autoscale_view()
        ax.set_title(label); ax.axis("off")
    cb = fig.colorbar(pc, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("chamfer distance mesh→scan (cm)")
    fig.suptitle(f"chamfer fit error  —  mean {dist_cm.mean():.2f} cm, "
                 f"p95 {vmax:.2f} cm", fontsize=11)
    plt.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"heatmap: {out_png}")
    return out_png


def overlay_final_mesh(fit_npz: Path, sapiens_dir: Path, capture_dir: Path,
                       height_cm: float, out_png: Path, *,
                       model_folder: str = "data/body_models") -> Path:
    """Overlay the final fitted mesh's silhouette on each capture photo.

    No registration — the mesh silhouette and the photo body silhouette
    are simply scaled to a common body height and centred, so the
    *shape* of the two outlines can be compared directly. Front view
    projects the mesh X-Y; side view projects Z-Y.
    """
    import smplx
    from PIL import Image

    from ..measure.regions import region_vertex_mask

    d = np.load(fit_npz)
    mv = d["smplx_vertices"].astype(np.float64)
    faces = smplx.create(model_path=model_folder, model_type="smplx",
                         gender=str(d["gender"]), num_betas=10,
                         use_pca=False, batch_size=1).faces.astype(np.int64)
    # Torso+legs only — the arm pose differs from the photo and would
    # clutter the silhouette comparison.
    keep = region_vertex_mask(("torso", "left_leg", "right_leg"),
                              model_folder=model_folder,
                              gender=str(d["gender"]))
    fmask = keep[faces].all(axis=1)
    tfaces = faces[fmask]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    # (image-x mesh axis, sign) per view.
    proj = {"front": (0, 1), "back": (0, -1),
            "side": (2, 1), "left": (2, 1), "right": (2, -1)}
    pngs = []
    for v in ("front", "back", "left", "right", "side"):
        photo = capture_dir / f"{v}.jpg"
        seg_p = sapiens_dir / f"{v}_seg.npy"
        if not photo.exists() or not seg_p.exists():
            continue
        ax0, sgn = proj[v]
        seg = np.load(seg_p)
        body = seg > 0
        img = Image.open(photo).convert("RGB").resize((seg.shape[1],
                                                       seg.shape[0]))
        rows = np.where(body.any(1))[0]
        cols = np.where(body.any(0))[0]
        crown, floor = rows.min(), rows.max()
        bcx = 0.5 * (cols.min() + cols.max())
        bpx = floor - crown

        u = sgn * mv[:, ax0]
        vv = mv[:, 1]
        scale = bpx / (vv.max() - vv.min())
        px = bcx + (u - np.median(u)) * scale
        py = crown + (vv.max() - vv) * scale
        P2 = np.column_stack([px, py])

        fig, a = plt.subplots(figsize=(seg.shape[1] / 110,
                                       seg.shape[0] / 110))
        a.imshow(np.asarray(img))
        a.add_collection(PolyCollection(
            P2[tfaces], facecolors="#39ff14", edgecolors="none", alpha=0.28))
        a.set_xlim(0, seg.shape[1]); a.set_ylim(seg.shape[0], 0)
        a.set_title(f"{v} — fitted mesh silhouette", fontsize=9)
        a.axis("off")
        plt.tight_layout()
        out = out_png.with_name(out_png.stem + f"_{v}.png")
        plt.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{v}] {out}")
        pngs.append(out)
    if pngs:
        _montage(pngs, out_png)
    return out_png


def _montage(pngs: list[Path], out_png: Path) -> None:
    """2×2 montage of the four view overlays."""
    from PIL import Image
    imgs = [Image.open(p) for p in pngs]
    cw = max(i.width for i in imgs)
    ch = max(i.height for i in imgs)
    sheet = Image.new("RGB", (cw * 2, ch * 2), "white")
    for k, im in enumerate(imgs):
        sheet.paste(im, ((k % 2) * cw, (k // 2) * ch))
    sheet.save(out_png)
    print(f"montage: {out_png}")


def main(argv: list[str] | None = None) -> int:
    """CLI: A-pose OBJ + per-view SMPL-X overlays for a 4-photo capture."""
    import argparse
    import json as _json

    p = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    p.add_argument("capture", type=Path,
                   help="Capture folder (front/back/left/right.jpg).")
    p.add_argument("--fit", type=Path, required=True,
                   help="Fitted SMPL-X npz (betas + gender).")
    p.add_argument("--sapiens-dir", type=Path, required=True,
                   help="Sapiens out/ dir with <view>.ply + <view>_seg.npy.")
    p.add_argument("--out-prefix", type=Path, required=True)
    p.add_argument("--height", type=float, default=None)
    p.add_argument("--model-folder", default="data/body_models")
    p.add_argument("--device", default="cpu")
    p.add_argument("--shoulder-deg", type=float, default=55.0,
                   help="A-pose arm drop; 55° = clear A (30° is ~T-pose).")
    args = p.parse_args(argv)

    height = args.height
    meta = args.capture / "capture_meta.json"
    if height is None and meta.exists():
        height = float(_json.loads(meta.read_text()).get("height_cm") or 0) \
            or None
    if height is None:
        raise SystemExit("no --height and no height_cm in capture_meta.json")

    # Part 1 — canonical A-pose OBJ.
    out_obj = args.out_prefix.with_name(args.out_prefix.name + "_apose.obj")
    repose_apose(args.fit, out_obj, height, shoulder_deg=args.shoulder_deg,
                 model_folder=args.model_folder)

    # Part 2 — per-view overlays in the original poses.
    d = np.load(args.fit)
    from .fit import FitConfig
    cfg = FitConfig(model_folder=args.model_folder, gender=str(d["gender"]),
                    num_betas=int(d["betas"].shape[0]), device=args.device,
                    partial_cloud=True, use_displacement=False)

    import smplx
    faces = smplx.create(model_path=args.model_folder, model_type="smplx",
                         gender=str(d["gender"]), num_betas=10,
                         use_pca=False, batch_size=1).faces.astype(np.int64)

    pngs = []
    for v in VIEWS:
        ply = args.sapiens_dir / f"{v}.ply"
        seg = args.sapiens_dir / f"{v}_seg.npy"
        # The original capture photo is already at the pointmap grid
        # resolution (w×h); the Sapiens out/<view>.jpg is a 2-panel vis.
        photo = args.capture / f"{v}.jpg"
        out_png = args.out_prefix.with_name(
            args.out_prefix.name + f"_overlay_{v}.png")
        overlay_view(v, ply, seg, photo, height, cfg, faces, out_png)
        pngs.append(out_png)

    _montage(pngs, args.out_prefix.with_name(
        args.out_prefix.name + "_overlays.png"))
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
