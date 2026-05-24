"""Composite canonical Anny mesh over photo silhouettes (front + side).

Orthographic projection of the mesh OBJ on top of the Sapiens body
silhouette. Mesh is scaled so its body-axis pixel height matches the
silhouette's body-axis pixel height, then centered on the silhouette
bbox center. Pure 2D — no perspective, no pose.

Use to visually QA the canonical shape produced by ``anny_photofit``
against the photo silhouettes without needing a pose-aware rasterizer.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_npz_mesh(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    return z["vertices"], z["faces"]


def body_bbox(seg_path: Path) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) tight bbox of body in segmentation."""
    seg = np.load(seg_path)
    if seg.ndim == 3:
        seg = seg.argmax(0) if seg.shape[0] < seg.shape[-1] else seg.argmax(-1)
    body = seg > 0
    ys, xs = np.where(body)
    return xs.min(), ys.min(), xs.max(), ys.max()


def project_ortho(verts: np.ndarray, view: str) -> np.ndarray:
    """Anny is Z-up: X=lateral, Y=depth, Z=vertical.
    Front view: image-X = +X, image-Y = -Z (flip so feet-down stays down).
    Side view : image-X = +Y, image-Y = -Z.
    """
    if view == "front":
        u = verts[:, 0]
        v = -verts[:, 2]
    else:  # side — photo subject faces +X in image (camera right).
        # Anny +Y is forward (away from face). Flip sign so the nose
        # points right in image, matching the photo orientation.
        u = -verts[:, 1]
        v = -verts[:, 2]
    return np.stack([u, v], axis=1)


def rasterize_triangles(uv: np.ndarray, faces: np.ndarray,
                         W: int, H: int) -> np.ndarray:
    """Scan-fill mesh triangles onto an HxW boolean mask. Used to build
    a 2D silhouette from the ortho-projected mesh."""
    from skimage.draw import polygon as skpoly
    mask = np.zeros((H, W), dtype=bool)
    # Pre-clip face indices to in-bounds verts to skip work.
    for tri in faces:
        try:
            pts = uv[tri]
        except IndexError:
            continue
        rr, cc = skpoly(pts[:, 1], pts[:, 0], shape=(H, W))
        if rr.size:
            mask[rr, cc] = True
    return mask


def compose(seg_path: Path, photo_path: Path, verts: np.ndarray,
            faces: np.ndarray, view: str, out_path: Path):
    img = np.array(Image.open(photo_path).convert("RGB"))
    H, W = img.shape[:2]
    x0, y0, x1, y1 = body_bbox(seg_path)
    body_h_px = (y1 - y0)
    body_cx = 0.5 * (x0 + x1)
    body_cy = 0.5 * (y0 + y1)

    uv = project_ortho(verts, view)
    mesh_h = uv[:, 1].max() - uv[:, 1].min()
    scale = body_h_px / max(mesh_h, 1e-6)
    uv = uv * scale
    uv[:, 0] += body_cx - 0.5 * (uv[:, 0].max() + uv[:, 0].min())
    uv[:, 1] += body_cy - 0.5 * (uv[:, 1].max() + uv[:, 1].min())

    silh = rasterize_triangles(uv, faces, W, H)
    # Overlay: 50% green tint on silhouette pixels.
    over = img.copy()
    tint = np.array([0, 220, 0], dtype=np.uint8)
    alpha = 0.45
    over[silh] = (over[silh] * (1 - alpha) + tint * alpha).astype(np.uint8)

    Image.fromarray(over).save(out_path)
    print(f"wrote {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fit-npz", type=Path, default=None,
                    help="Single mesh NPZ used for both views.")
    ap.add_argument("--front-npz", type=Path, default=None,
                    help="Mesh NPZ for front view (overrides --fit-npz).")
    ap.add_argument("--side-npz", type=Path, default=None,
                    help="Mesh NPZ for side view (overrides --fit-npz).")
    ap.add_argument("--front-photo", type=Path,
                    default=Path("data/captures/me_photos/pair_20260522_090554/front.jpg"))
    ap.add_argument("--side-photo", type=Path,
                    default=Path("data/captures/me_photos/pair_20260522_090554/side.jpg"))
    ap.add_argument("--front-seg", type=Path,
                    default=Path("data/results/pair_20260522_090554_sapiens/out/front_seg.npy"))
    ap.add_argument("--side-seg", type=Path,
                    default=Path("data/results/pair_20260522_090554_sapiens/out/side_seg.npy"))
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args(argv)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    front_path = args.front_npz or args.fit_npz
    side_path = args.side_npz or args.fit_npz
    if front_path is None or side_path is None:
        raise SystemExit("Provide --fit-npz or both --front-npz/--side-npz.")
    fverts, ffaces = load_npz_mesh(front_path)
    sverts, sfaces = load_npz_mesh(side_path)
    compose(args.front_seg, args.front_photo, fverts, ffaces, "front",
            args.out_prefix.with_name(args.out_prefix.name + "_front.png"))
    compose(args.side_seg, args.side_photo, sverts, sfaces, "side",
            args.out_prefix.with_name(args.out_prefix.name + "_side.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
