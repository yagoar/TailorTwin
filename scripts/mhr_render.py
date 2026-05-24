"""Render Multi-HMR's RAW v3d (camera-space mesh) over the input
photo via perspective projection. No reshaping, no tape, no my pose.

Lets us see what shape Multi-HMR actually predicts for the person —
if THAT matches the photo silhouette closely, our parametric Anny
refit is the lossy step and we should either:
  a) use MHR's mesh directly as the production output, or
  b) re-fit Anny to MHR's v3d (vert-to-vert), not just its phenotype.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Multi-HMR pipeline center-crops input to a square, then resizes to
# 672 (the model's img_size). The photo here is 865x1440 (portrait):
# crop size = min(865, 1440) = 865, vertical center crop.
MHR_IMG_SIZE = 672


def load_mhr(path: Path) -> dict:
    return torch.load(path, weights_only=False, map_location="cpu")


def render_view(pt_path: Path, photo_path: Path, faces_path: Path | None,
                view: str, out_path: Path):
    h = load_mhr(pt_path)
    v3d = h["v3d"].cpu().numpy()   # (N, 3) camera space
    K = h["K"].cpu().numpy()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Perspective project.
    u = fx * v3d[:, 0] / v3d[:, 2] + cx
    v = fy * v3d[:, 1] / v3d[:, 2] + cy
    print(f"{view}: u range {u.min():.0f}..{u.max():.0f}  "
          f"v range {v.min():.0f}..{v.max():.0f}  (img_size {MHR_IMG_SIZE})")

    # Photo as Multi-HMR saw it: center-cropped square + resized 672.
    img = Image.open(photo_path).convert("RGB")
    W, H = img.size
    s = min(W, H)
    left, top = (W - s) // 2, (H - s) // 2
    img_c = img.crop((left, top, left + s, top + s)).resize(
        (MHR_IMG_SIZE, MHR_IMG_SIZE))
    arr = np.array(img_c)

    # Rasterize as silhouette via skimage triangle fill if faces avail.
    if faces_path and faces_path.exists():
        faces = np.load(faces_path)
    else:
        # Try recovering Anny faces (Multi-HMR's Anny variant uses
        # remove_unattached_vertices=False → 19158 verts, 27420 faces).
        import anny
        m = anny.create_fullbody_model(remove_unattached_vertices=False)
        faces = m.get_triangular_faces().cpu().numpy()

    from skimage.draw import polygon as skpoly
    silh = np.zeros((MHR_IMG_SIZE, MHR_IMG_SIZE), dtype=bool)
    uv = np.stack([u, v], axis=1)
    for tri in faces:
        if tri.max() >= len(uv):
            continue
        pts = uv[tri]
        rr, cc = skpoly(pts[:, 1], pts[:, 0], shape=silh.shape)
        if rr.size:
            silh[rr, cc] = True

    over = arr.copy()
    tint = np.array([0, 220, 0], dtype=np.uint8)
    alpha = 0.45
    over[silh] = (over[silh] * (1 - alpha) + tint * alpha).astype(np.uint8)
    Image.fromarray(over).save(out_path)
    print(f"wrote {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--front-pt", type=Path,
                    default=Path("data/results/multihmr_spike/our_out_front_p0.pt"))
    ap.add_argument("--side-pt", type=Path,
                    default=Path("data/results/multihmr_spike/our_out_side_p0.pt"))
    ap.add_argument("--front-photo", type=Path,
                    default=Path("data/captures/me_photos/pair_20260522_090554/front.jpg"))
    ap.add_argument("--side-photo", type=Path,
                    default=Path("data/captures/me_photos/pair_20260522_090554/side.jpg"))
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    render_view(args.front_pt, args.front_photo, None, "front",
                args.out_prefix.with_name(args.out_prefix.name + "_front.png"))
    render_view(args.side_pt,  args.side_photo,  None, "side",
                args.out_prefix.with_name(args.out_prefix.name + "_side.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
