"""Side-silhouette seat correction of a front-depth fit.

The front Sapiens pointmap fit pins the front body surface — bust,
belly, chest, width — metrically. The buttock back is unseen by a front
shot, so the seat depth comes from the SMPL-X prior and the hip reads
low.

The side **photo** sees the seat. Its silhouette is reliable for the
seat's *back* extent (the down arm sits inside the outline; the back is
never occluded). This stage pushes the buttock back out to that depth
over a wide band — waist to mid-thigh, torso and legs together so the
buttock does not seam at the region boundary — with the correction
cosine-tapered to 1 at the band ends and clamped so no vertex is flung
forward of the front surface.

Only the seat is corrected. A full side-profile match was tried: the
side photo is not a perfect 90° turn, so its absolute depths are not
metrically trustworthy, and reshaping the whole torso to them inflates
the waist/highhip. The front pointmap stays the metric authority above
the waist; the side fixes only the one thing the front shot cannot see.

Two meshes are written: the fit-pose npz the measure pipeline reads (a
canonical A-pose rebuild breaks several measure landmarks) and a clean
A-pose OBJ.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

N = 40
SEAT_LO, SEAT_HI = 0.34, 0.68


def _smooth(a, k=5):
    out = a.copy()
    pad = np.pad(a, k // 2, mode="edge")
    for i in range(len(a)):
        w = pad[i:i + k]
        out[i] = np.nanmean(w) if np.any(np.isfinite(w)) else np.nan
    return out


def _extend_back(verts, region, side_prof, y_lo, y_hi, *, n=N):
    """Push the back of each seat slice out to the side-photo depth.

    Front Z held (front pointmap fixed it); back rescaled so total depth
    matches the side silhouette. The ratio is cosine-tapered to 1 over
    the outer 25 % of the band, only band-Y vertices are touched, and
    the result is clamped to ``z_front`` so a belly vertex forward of
    the percentile front line cannot be flung into a spike."""
    from .silhouette import sample_extent_cm, sample_valid

    v = verts.copy()
    y = v[:, 1]
    ymin, span = float(y.min()), float(y.max() - y.min())
    yn = np.linspace(y_lo, y_hi, n)
    band = 0.7 * (y_hi - y_lo) / (n - 1)
    zf = np.full(n, np.nan)
    ratio = np.full(n, np.nan)
    for i, g in enumerate(yn):
        yw = ymin + g * span
        sel = region & (np.abs(y - yw) < band * span)
        if sel.sum() < 8 or not sample_valid(side_prof, g):
            continue
        z = v[sel, 2]
        z_front = np.percentile(z, 99)
        cur = z_front - np.percentile(z, 1)
        tgt = sample_extent_cm(side_prof, g) / 100.0
        if cur > 1e-3:
            zf[i] = z_front
            ratio[i] = np.clip(tgt / cur, 0.85, 1.6)
    ratio = _smooth(ratio)
    ok = np.isfinite(ratio) & np.isfinite(zf)
    if ok.sum() < 4:
        return v
    f = np.linspace(0.0, 1.0, n)
    w = np.clip(np.minimum(f, 1.0 - f) / 0.25, 0.0, 1.0)
    w = 0.5 - 0.5 * np.cos(np.pi * w)
    ratio = 1.0 + (ratio - 1.0) * w
    ya = ymin + yn[ok] * span
    yl = v[:, 1]
    zf_v = np.interp(yl, ya, zf[ok])
    r_v = np.interp(yl, ya, ratio[ok], left=1.0, right=1.0)
    m = region & (yl >= ya[0]) & (yl <= ya[-1])
    znew = zf_v[m] - (zf_v[m] - v[m, 2]) * r_v[m]
    v[m, 2] = np.minimum(znew, zf_v[m])
    return v


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("front_fit", type=Path)
    p.add_argument("side_seg", type=Path)
    p.add_argument("--height", type=float, required=True)
    p.add_argument("--out-prefix", type=Path, required=True)
    p.add_argument("--model-folder", default="data/body_models")
    p.add_argument("--a-pose-shoulder-deg", type=float, default=40.0)
    args = p.parse_args(argv)

    import smplx
    import torch

    from .refine_to_tape import _build_a_pose
    from .silhouette import extract_profile, load_silhouette
    from .silhouette_transfer import _laplacian_relax
    from ..measure.exports import write_obj
    from ..measure.regions import region_vertex_mask
    from ..fit.fit import fit_gender

    fit = np.load(args.front_fit)
    gender = fit_gender(fit)
    betas = fit["betas"].astype(np.float32)
    num_betas = int(betas.shape[0])
    bm = smplx.create(model_path=args.model_folder, model_type="smplx",
                      gender=gender, num_betas=num_betas, use_pca=False,
                      flat_hand_mean=True, batch_size=1)
    faces = bm.faces.astype(np.int64)
    lower = region_vertex_mask(("torso", "left_leg", "right_leg"),
                               model_folder=args.model_folder,
                               gender=gender)

    side_mask, _ = load_silhouette(str(args.side_seg), arm_classes=())
    side_prof = extract_profile(side_mask, args.height, view="side")

    def correct(verts, *, rescale):
        v = _extend_back(verts, lower, side_prof, SEAT_LO, SEAT_HI)
        v = _laplacian_relax(v, faces, lower, iters=4, lam=0.2)
        if rescale:
            ymin = float(v[:, 1].min())
            cur = (v[:, 1].max() - ymin) * 100.0
            v[:, 1] = (v[:, 1] - ymin) * (args.height / cur)
        return v

    verts = correct(fit["smplx_vertices"].astype(np.float64), rescale=False)

    a_pose = _build_a_pose(args.a_pose_shoulder_deg).astype(np.float32)
    with torch.no_grad():
        out = bm(betas=torch.from_numpy(betas)[None],
                 body_pose=torch.from_numpy(a_pose.reshape(1, -1)),
                 global_orient=torch.zeros(1, 3), transl=torch.zeros(1, 3))
    apose_v = correct(out.vertices[0].numpy().astype(np.float64),
                      rescale=True)

    out_npz = args.out_prefix.with_name(args.out_prefix.name
                                        + "_smplx_fit.npz")
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: fit[k] for k in fit.files}
    payload["smplx_vertices"] = verts.astype(np.float32)
    np.savez(out_npz, **payload)
    out_apose = args.out_prefix.with_name(args.out_prefix.name
                                          + "_apose.obj")
    write_obj(apose_v.astype(np.float32), faces, out_apose)
    print(f"wrote {out_npz}\nwrote {out_apose}  (A-pose, visual)")

    out_csv = args.out_prefix.with_name(args.out_prefix.name
                                        + "_measurements.csv")
    out_obj = args.out_prefix.with_name(args.out_prefix.name
                                        + "_fit_body.obj")
    cmd = [sys.executable, "-m", "tailor_twin.measure.cli", str(out_npz),
           "--num-betas", str(num_betas), "--gender", gender,
           "--model-folder", args.model_folder,
           "--save-csv", str(out_csv), "--save-obj", str(out_obj)]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
