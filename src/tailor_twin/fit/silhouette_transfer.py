"""Direct silhouette width transfer — photo outlines → SMPL-X widths.

The pipeline the user asked for:

1. Sapiens fixes the per-view pose (``pose_fit``) — recorded for the
   posed-mesh overlay; the width transfer itself is pose-agnostic
   because it works on normalized body height.
2. The front photo gives torso/leg **width** at every height and the
   side photo gives **depth**; each canonical SMPL-X slice is scaled in
   X and Z to those exact photo extents (``apply_anisotropic_profile``).
   Nothing is removed from the side silhouette — the down arm is inside
   the body outline, so the outline is already correct.
3. A light Laplacian relaxation removes any per-slice dents, then every
   girth is measured on the resulting mesh.

Unlike ``silhouette_betas`` (which solves CAESAR betas and so pulls each
girth toward the population mean — the observed waist under-read), this
scales the mesh straight onto the measured outline: widths come out
exact, the cross-section shape is the model's, the girth follows.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

N_SLICES = 26
TORSO_LO, TORSO_HI = 0.46, 0.83      # feet-fraction band of the torso
LEG_LO, LEG_HI = 0.06, 0.46


def _mesh_extents(verts, region_mask, y_norm, band):
    """Per-slice X width and Z depth (cm) of a mesh region."""
    from .silhouette_betas import _slice_extents
    w, d, _ = _slice_extents(verts, region_mask, y_norm, band)
    return w, d


def _scale_curve(photo_cm, mesh_cm, valid):
    """photo / mesh ratio per slice; NaN where unmeasured (clamped)."""
    s = np.full(len(photo_cm), np.nan)
    ok = valid & np.isfinite(photo_cm) & np.isfinite(mesh_cm) & (mesh_cm > 1)
    s[ok] = np.clip(photo_cm[ok] / mesh_cm[ok], 0.6, 1.6)
    return s


def transfer(
    front_seg: str, side_seg: str, *, height_cm: float,
    base_betas: np.ndarray, gender: str,
    model_folder: str = "data/body_models",
    a_pose_shoulder_deg: float = 30.0,
) -> dict:
    """Scale the canonical SMPL-X mesh onto the two photo silhouettes."""
    import smplx
    import torch

    from .refine_to_tape import _build_a_pose
    from .ring_deform import apply_anisotropic_profile
    from .silhouette import extract_profile, load_silhouette, sample_extent_cm, sample_valid
    from ..measure.regions import region_vertex_mask

    # --- photo profiles -------------------------------------------------
    # Front: arms are out and separable, so the arm classes are dropped.
    # Side: keep the whole silhouette — the down arm is interior to the
    # body outline, removing it would only punch holes.
    front_mask, _ = load_silhouette(front_seg)                   # arm-free
    side_mask, _ = load_silhouette(side_seg, arm_classes=())     # full body
    front_prof = extract_profile(front_mask, height_cm, view="front",
                                 arm_free=True)
    side_prof = extract_profile(side_mask, height_cm, view="side")
    front_leg = extract_profile(front_mask, height_cm, view="front",
                                pick="widest", arm_free=True)
    side_leg = extract_profile(side_mask, height_cm, view="side",
                               pick="widest")

    # --- canonical mesh -------------------------------------------------
    num_betas = base_betas.shape[0]
    bm = smplx.create(model_path=model_folder, model_type="smplx",
                      gender=gender, num_betas=num_betas, use_pca=False,
                      flat_hand_mean=True, batch_size=1)
    pose = _build_a_pose(a_pose_shoulder_deg).astype(np.float32)
    with torch.no_grad():
        out = bm(betas=torch.from_numpy(base_betas[None].astype(np.float32)),
                 body_pose=torch.from_numpy(pose.reshape(1, -1)),
                 global_orient=torch.zeros(1, 3), transl=torch.zeros(1, 3))
    verts = out.vertices[0].numpy().astype(np.float64)
    y = verts[:, 1]
    y_min, span = float(y.min()), float(y.max() - y.min())

    torso_mask = region_vertex_mask(("torso",), model_folder=model_folder,
                                    gender=gender)
    lleg = region_vertex_mask(("left_leg",), model_folder=model_folder,
                              gender=gender)
    rleg = region_vertex_mask(("right_leg",), model_folder=model_folder,
                              gender=gender)

    deformed = verts.copy()
    report = {}
    # measure_mask = where the slice extents are read; apply_masks = the
    # regions the scale curve is applied to (legs: one leg measured, both
    # scaled — each about its own centroid so they keep their stance).
    for name, measure_mask, apply_masks, lo, hi, fp, sp in (
            ("torso", torso_mask, [torso_mask], TORSO_LO, TORSO_HI,
             front_prof, side_prof),
            ("legs", lleg, [lleg, rleg], LEG_LO, LEG_HI,
             front_leg, side_leg)):
        yn = np.linspace(lo, hi, N_SLICES)
        band = 0.5 * (hi - lo) / (N_SLICES - 1)
        mw, md = _mesh_extents(deformed, measure_mask, yn, band)
        pw = np.array([sample_extent_cm(fp, g) for g in yn])
        pd = np.array([sample_extent_cm(sp, g) for g in yn])
        vw = np.array([sample_valid(fp, g) for g in yn])
        vd = np.array([sample_valid(sp, g) for g in yn])
        sx = _scale_curve(pw, mw, vw)
        sz = _scale_curve(pd, md, vd)
        y_levels = y_min + yn * span
        for am in apply_masks:
            deformed = apply_anisotropic_profile(deformed, y_levels, sx, sz,
                                                 region_mask=am)
        report[name] = dict(y_norm=yn, photo_w=pw, photo_d=pd,
                            mesh_w=mw, mesh_d=md, sx=sx, sz=sz)

    # --- light Laplacian relaxation (de-dent) ---------------------------
    deformed = _laplacian_relax(deformed, bm.faces.astype(np.int64),
                                torso_mask | lleg | rleg, iters=6, lam=0.25)

    # --- height to exact stature ---------------------------------------
    ym = float(deformed[:, 1].min())
    cur_h = (deformed[:, 1].max() - ym) * 100.0
    deformed[:, 1] = ym + (deformed[:, 1] - ym) * (height_cm / cur_h)

    return {"verts": deformed.astype(np.float32),
            "joints": out.joints[0].numpy().astype(np.float32),
            "body_pose": pose, "betas": base_betas.astype(np.float32),
            "report": report}


def _laplacian_relax(verts, faces, region, *, iters, lam):
    """Tangential Laplacian smoothing on a region — removes the slice
    dents the per-Y scaling can leave, without shrinking girth (the
    update is the average of neighbours, applied only inside ``region``
    and damped by ``lam``)."""
    nv = verts.shape[0]
    nbr = [[] for _ in range(nv)]
    for a, b, c in faces:
        nbr[a] += [b, c]; nbr[b] += [a, c]; nbr[c] += [a, b]
    nbr = [np.unique(n) for n in nbr]
    v = verts.copy()
    for _ in range(iters):
        upd = v.copy()
        for i in np.where(region)[0]:
            if len(nbr[i]):
                upd[i] = v[i] + lam * (v[nbr[i]].mean(0) - v[i])
        v = upd
    return v


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("front", type=Path, help="front Sapiens '*_seg.npy'")
    p.add_argument("side", type=Path, help="side Sapiens '*_seg.npy'")
    p.add_argument("--height", type=float, required=True)
    p.add_argument("--base-fit", type=Path, required=True)
    p.add_argument("--out-prefix", type=Path, required=True)
    p.add_argument("--model-folder", default="data/body_models")
    p.add_argument("--a-pose-shoulder-deg", type=float, default=30.0)
    args = p.parse_args(argv)

    fit = np.load(args.base_fit)
    from ..fit.fit import fit_gender
    gender = fit_gender(fit)
    base_betas = fit["betas"].astype(np.float64)

    res = transfer(str(args.front), str(args.side), height_cm=args.height,
                   base_betas=base_betas, gender=gender,
                   model_folder=args.model_folder,
                   a_pose_shoulder_deg=args.a_pose_shoulder_deg)

    for nm, r in res["report"].items():
        print(f"\n{nm} — photo vs canonical mesh (cm):")
        print(f"  {'ynorm':>6} {'Wphoto':>7} {'Wmesh':>7} {'sx':>5}"
              f"  {'Dphoto':>7} {'Dmesh':>7} {'sz':>5}")
        for i in range(0, N_SLICES, 3):
            print(f"  {r['y_norm'][i]:6.2f} {r['photo_w'][i]:7.1f} "
                  f"{r['mesh_w'][i]:7.1f} {r['sx'][i]:5.2f}  "
                  f"{r['photo_d'][i]:7.1f} {r['mesh_d'][i]:7.1f} "
                  f"{r['sz'][i]:5.2f}")

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
    print(f"\nwrote {out_npz}")

    out_csv = out_prefix.with_name(out_prefix.name + "_measurements.csv")
    out_obj = out_prefix.with_name(out_prefix.name + "_fit_body.obj")
    cmd = [sys.executable, "-m", "tailor_twin.measure.cli", str(out_npz),
           "--num-betas", str(base_betas.shape[0]), "--gender", gender,
           "--model-folder", args.model_folder,
           "--save-csv", str(out_csv), "--save-obj", str(out_obj)]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
