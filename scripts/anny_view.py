"""Render the Anny mesh from canonical front/side/3-quarter angles
WITHOUT photo background — pure mesh visualization. Use to inspect
shape independently of photo alignment / projection ambiguity.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.draw import polygon as skpoly


def project(verts: np.ndarray, view: str) -> np.ndarray:
    # Auto-detect vertical axis (largest span).
    spans = [verts[:, i].max() - verts[:, i].min() for i in range(3)]
    vert_axis = int(np.argmax(spans))   # axis aligned with body height
    horiz_axes = [i for i in range(3) if i != vert_axis]
    # Pick which horizontal is "lateral" (typically larger of the two).
    h_spans = [spans[i] for i in horiz_axes]
    lat_axis = horiz_axes[int(np.argmax(h_spans))]
    depth_axis = [i for i in horiz_axes if i != lat_axis][0]
    if view == "front":
        u, v = verts[:, lat_axis], -verts[:, vert_axis]
    elif view == "side":
        u, v = -verts[:, depth_axis], -verts[:, vert_axis]
    elif view == "back":
        u, v = -verts[:, lat_axis], -verts[:, vert_axis]
    elif view == "3q":
        rot = np.array([[np.cos(np.deg2rad(45)), np.sin(np.deg2rad(45))],
                        [-np.sin(np.deg2rad(45)), np.cos(np.deg2rad(45))]])
        xy = verts[:, [lat_axis, depth_axis]] @ rot
        u, v = xy[:, 0], -verts[:, vert_axis]
    return np.stack([u, v], axis=1)


def shade(verts: np.ndarray, faces: np.ndarray, view: str,
          W: int = 400, H: int = 700) -> np.ndarray:
    uv = project(verts, view)
    pad = 0.05
    mn, mx = uv.min(0), uv.max(0)
    scale = min((W * (1 - 2*pad)) / (mx[0] - mn[0]),
                (H * (1 - 2*pad)) / (mx[1] - mn[1]))
    uv = uv * scale
    uv -= uv.mean(0)
    uv[:, 0] += W / 2
    uv[:, 1] += H / 2

    # Per-vertex depth (axis facing away from viewer).
    spans = [verts[:, i].max() - verts[:, i].min() for i in range(3)]
    vert_axis = int(np.argmax(spans))
    horiz_axes = [i for i in range(3) if i != vert_axis]
    h_spans = [spans[i] for i in horiz_axes]
    lat_axis = horiz_axes[int(np.argmax(h_spans))]
    depth_axis = [i for i in horiz_axes if i != lat_axis][0]
    if view == "front":
        z = verts[:, depth_axis]
    elif view == "side":
        z = verts[:, lat_axis]
    elif view == "back":
        z = -verts[:, depth_axis]
    else:
        z = verts[:, lat_axis] - verts[:, depth_axis]
    z = (z - z.min()) / (z.max() - z.min() + 1e-9)

    canvas = np.full((H, W, 3), 240, dtype=np.uint8)
    depth = np.full((H, W), 1e9, dtype=np.float32)

    for tri in faces:
        if tri.max() >= len(uv): continue
        pts = uv[tri]
        zt = z[tri].mean()
        if zt > 0.85: continue  # back-face cull approx
        rr, cc = skpoly(pts[:, 1], pts[:, 0], shape=(H, W))
        if not rr.size: continue
        mask = depth[rr, cc] > zt
        # Cool greenish shade scaled by depth (front = brighter).
        shade = 1.0 - zt
        col = np.array([int(60 + 80*shade), int(200 - 50*zt), int(60 + 80*shade)],
                       dtype=np.uint8)
        canvas[rr[mask], cc[mask]] = col
        depth[rr[mask], cc[mask]] = zt
    return canvas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fit-npz", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args(argv)
    z = np.load(args.fit_npz, allow_pickle=True)
    verts, faces = z["vertices"], z["faces"]
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for view in ("front", "side", "back", "3q"):
        img = shade(verts, faces, view)
        out = args.out_prefix.with_name(args.out_prefix.name + f"_{view}.png")
        Image.fromarray(img).save(out)
        print(f"wrote {out}")
    # Side-by-side composite
    imgs = [shade(verts, faces, v) for v in ("front", "3q", "side", "back")]
    full = np.concatenate(imgs, axis=1)
    out = args.out_prefix.with_name(args.out_prefix.name + "_grid.png")
    Image.fromarray(full).save(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
