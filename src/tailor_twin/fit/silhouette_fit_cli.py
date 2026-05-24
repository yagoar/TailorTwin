"""CLI: fit SMPL-X shape to two silhouette photos (3DLook-style).

This is the offline reconstruction of 3DLook's Mobile Tailor geometry.
Two photos — front and side, tight clothing, phone held vertical — plus
the user's height give the body's true outline:

  * front photo → body **width** at every height,
  * side photo  → body **depth** at every height.

``silhouette.extract_profile`` measures those curves. This CLI then
solves the SMPL-X **shape betas** whose A-pose mesh reproduces them
(``silhouette_betas.fit_betas_to_silhouette``).

Why betas, not free-form deformation
------------------------------------
Scaling vertices per slice to hit the silhouette exactly produces a body
that is off the shape manifold — dented, over-pinched, seamed at the
limbs. Fitting betas keeps every iterate inside the CAESAR-trained
SMPL-X shape space, so the result is always an anatomically plausible
body. This is the approach the literature converges on (Škorvánková et
al. arXiv:2205.14347; Ruiz et al. BMnet arXiv:2210.05667). The trade-off
is that an arbitrary silhouette may leave a 1-2 cm residual on some
girths — run ``ring-deform`` afterwards if you need tape-exact numbers.

The base SMPL-X comes from ``--base-fit`` (betas seed + gender). Topology
is untouched, so the measurement extractor, ``ring-deform`` and the
garment pipeline all keep working on the result.

Example::

    tailor-twin silhouette-fit front_seg.npy side_seg.npy \\
        --height 160 --base-fit data/results/me_shapy_smplx_fit.npz \\
        --out-prefix data/results/me_silhouette

Capture tips
------------
* Front photo: arms held ~45° out from the torso.
* Side photo: a true 90° profile; arms down at the sides.
* Phone vertical, subject filling the frame head-to-foot, plain
  background, even lighting. The ``tailor-twin capture`` webapp enforces
  the vertical-phone constraint via the gyroscope.
* Prefer Sapiens part-seg ``*_seg.npy`` inputs — the arms are removed
  cleanly, which a raw photo cannot do for the side view.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("front", type=Path,
                   help="Front view: a Sapiens part-seg '*_seg.npy' "
                        "(arms auto-removed — preferred) or an RGB photo.")
    p.add_argument("side", type=Path,
                   help="Side view (90°): a '*_seg.npy' or an RGB photo.")
    p.add_argument("--height", type=float, required=True,
                   help="Subject height in cm (absolute scale reference).")
    p.add_argument("--base-fit", type=Path, required=True,
                   help="Existing fit npz — seeds betas + supplies gender.")
    p.add_argument("--out-prefix", type=Path, required=True)
    p.add_argument("--model-folder", default="data/body_models")
    p.add_argument("--seg-backend", default="rvm",
                   choices=["rvm", "rembg"],
                   help="Photo segmentation backend (raw-photo inputs only).")
    p.add_argument("--n-active", type=int, default=20,
                   help="Number of leading SMPL-X betas to optimize.")
    p.add_argument("--n-slices", type=int, default=24,
                   help="Torso control slices between --y-lo and --y-hi.")
    p.add_argument("--y-lo", type=float, default=0.50,
                   help="Lowest torso slice, fraction of body height "
                        "(0=feet, 1=crown). ~0.50 ≈ crotch.")
    p.add_argument("--y-hi", type=float, default=0.82,
                   help="Highest torso slice, fraction of body height. "
                        "~0.82 ≈ shoulder.")
    p.add_argument("--max-iters", type=int, default=18)
    p.add_argument("--target", action="append", default=[],
                   help="CODE=value_cm tape target (repeatable). When set, "
                        "ring-deform polishes the betas fit to hit these "
                        "exactly — photos give shape, tape gives size. "
                        "Output: <out-prefix>_taped_*.")
    p.add_argument("--fit-arms", action="store_true",
                   help="Also fit arm width (needs a Sapiens '*_seg.npy' "
                        "front input). Off by default — arm betas compete "
                        "with the torso/leg betas and can degrade the "
                        "overall fit; the base arm is usually within ~1 cm.")
    p.add_argument("--a-pose-shoulder-deg", type=float, default=30.0,
                   help="Canonical A-pose shoulder angle (0 = T-pose).")
    args = p.parse_args(argv)

    for ph in (args.front, args.side, args.base_fit):
        if not ph.exists():
            raise SystemExit(f"missing input: {ph}")

    from .silhouette import extract_profile, load_silhouette

    print(f"loading front silhouette ({args.front.name}) …")
    front_mask, front_arm_free = load_silhouette(
        str(args.front), backend=args.seg_backend)
    print(f"loading side silhouette ({args.side.name}) …")
    side_mask, side_arm_free = load_silhouette(
        str(args.side), backend=args.seg_backend)
    if not side_arm_free:
        print("  WARNING: side view is a raw photo — arms inflate the "
              "depth profile. Pass a Sapiens '*_seg.npy' for accuracy.")

    front_prof = extract_profile(front_mask, args.height, view="front",
                                 arm_free=front_arm_free)
    side_prof = extract_profile(side_mask, args.height, view="side",
                                arm_free=side_arm_free)
    # Leg profiles: below the crotch each scan-line splits into two leg
    # blobs — pick="widest" reads a single leg (width from the front,
    # depth from the side). Drives the thigh/calf part of the fit.
    front_leg_prof = extract_profile(front_mask, args.height, view="front",
                                     pick="widest")
    side_leg_prof = extract_profile(side_mask, args.height, view="side",
                                    pick="widest")
    print(f"front silhouette: {front_prof.height_px:.0f} px tall")
    print(f"side  silhouette: {side_prof.height_px:.0f} px tall")

    # Arm width profile — opt-in (--fit-arms). Needs a Sapiens part-seg
    # to isolate the arm pixels; a raw RVM matte cannot separate arm
    # from torso.
    arm_prof = None
    if args.fit_arms:
        if str(args.front).endswith(".npy"):
            from .silhouette import (SAPIENS_ARM_CLASSES,
                                     arm_profile_from_seg)
            seg = np.load(args.front)
            counts = {c: int((seg == c).sum()) for c in SAPIENS_ARM_CLASSES}
            arm_class = max(counts, key=counts.get)
            arm_prof = arm_profile_from_seg(seg, args.height,
                                            arm_class=arm_class)
            ok = int(np.sum(np.isfinite(arm_prof)))
            print(f"arm profile: seg class {arm_class}, {ok}/"
                  f"{len(arm_prof)} valid bins")
        else:
            print("  --fit-arms ignored: front is a raw photo, no part-seg "
                  "to isolate the arm. Arm shape stays at the base.")

    fit = np.load(args.base_fit)
    from ..fit.fit import fit_gender
    gender = fit_gender(fit)
    base_betas = fit["betas"].astype(np.float64)
    num_betas = base_betas.shape[0]

    # --- parametric core: solve betas so the mesh matches the photos ---
    from .silhouette_betas import fit_betas_to_silhouette
    result = fit_betas_to_silhouette(
        base_betas, gender, front_prof, side_prof,
        height_cm=args.height,
        front_leg_prof=front_leg_prof, side_leg_prof=side_leg_prof,
        arm_prof=arm_prof,
        model_folder=args.model_folder,
        a_pose_shoulder_deg=args.a_pose_shoulder_deg,
        n_active=args.n_active, n_slices=args.n_slices,
        y_lo=args.y_lo, y_hi=args.y_hi, max_iters=args.max_iters)

    # Per-slice match report (cm).
    print("\ntorso slices — silhouette target vs fitted mesh (cm):")
    print(f"  {'Y':>5}  {'width tgt/got':>16}  {'depth tgt/got':>16}")
    step = max(args.n_slices // 8, 1)
    for i in range(0, args.n_slices, step):
        print(f"  {result.y_norm[i]:5.2f}  "
              f"{result.width_target[i]:7.1f}/{result.width_after[i]:7.1f}  "
              f"{result.depth_target[i]:7.1f}/{result.depth_after[i]:7.1f}")

    if result.leg_y_norm is not None:
        print("\nleg slices — silhouette target vs fitted mesh (cm):")
        print(f"  {'Y':>5}  {'width tgt/got':>16}  {'depth tgt/got':>16}")
        nleg = len(result.leg_y_norm)
        for i in range(0, nleg, max(nleg // 6, 1)):
            print(f"  {result.leg_y_norm[i]:5.2f}  "
                  f"{result.leg_width_target[i]:7.1f}/"
                  f"{result.leg_width_after[i]:7.1f}  "
                  f"{result.leg_depth_target[i]:7.1f}/"
                  f"{result.leg_depth_after[i]:7.1f}")

    if result.arm_width_target is not None:
        print("\narm width — silhouette target vs fitted mesh (cm, "
              "shoulder→wrist):")
        for i in range(len(result.arm_width_target)):
            t = result.arm_width_target[i]
            g = result.arm_width_after[i]
            ts = f"{t:6.1f}" if np.isfinite(t) else "    --"
            gs = f"{g:6.1f}" if np.isfinite(g) else "    --"
            print(f"  bin {i}  tgt {ts}  got {gs}")

    # --- rebuild final mesh from fitted betas, scale to height ---
    import smplx
    import torch

    from .refine_to_tape import _build_a_pose
    bm = smplx.create(model_path=args.model_folder, model_type="smplx",
                      gender=gender, num_betas=num_betas,
                      use_pca=False, flat_hand_mean=True, batch_size=1)
    canon_pose = _build_a_pose(args.a_pose_shoulder_deg).astype(np.float32)
    with torch.no_grad():
        out = bm(
            betas=torch.from_numpy(result.betas[None, :].astype(np.float32)),
            body_pose=torch.from_numpy(canon_pose.reshape(1, -1)),
            global_orient=torch.zeros(1, 3),
            transl=torch.zeros(1, 3),
            return_full_pose=False,
        )
    verts = out.vertices[0].cpu().numpy().astype(np.float64)
    joints = out.joints[0].cpu().numpy().astype(np.float32)

    # Uniform Y-scale so feet→crown equals the user's height. This is a
    # pure stretch along Y — it leaves every X/Z silhouette extent (and
    # therefore every girth the betas fit just matched) untouched.
    y_min = float(verts[:, 1].min())
    cur_h_cm = (verts[:, 1].max() - y_min) * 100.0
    s_h = args.height / cur_h_cm
    verts[:, 1] = y_min + (verts[:, 1] - y_min) * s_h
    print(f"\nheight: {cur_h_cm:.1f} → ×{s_h:.4f} → {args.height} cm")

    # Save a fit npz mirroring the base with the fitted betas + mesh.
    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_npz = out_prefix.with_name(out_prefix.name + "_smplx_fit.npz")
    payload = {k: fit[k] for k in fit.files}
    payload["betas"] = result.betas.astype(np.float32)
    payload["smplx_vertices"] = verts.astype(np.float32)
    payload["smplx_joints"] = joints
    payload["body_pose"] = canon_pose
    payload["global_orient"] = np.zeros((3,), dtype=np.float32)
    payload["transl"] = np.zeros((3,), dtype=np.float32)
    payload["z"] = np.array([])
    np.savez(out_npz, **payload)
    print(f"wrote {out_npz}")

    # Tape polish. The silhouette pins body *shape*; a single side photo
    # cannot guarantee a true 90° turn, so the absolute depth (and thus
    # the girths) can drift a few cm. When tape targets are supplied,
    # ring-deform nudges the betas-fit geometry to hit them exactly —
    # photos give the proportions, tape gives the size.
    if args.target:
        taped = out_prefix.with_name(out_prefix.name + "_taped")
        cmd = [sys.executable, "-m", "tailor_twin.fit.ring_deform_cli",
               str(out_npz), "--out-prefix", str(taped),
               "--model-folder", args.model_folder]
        for t in args.target:
            cmd += ["--target", t]
        print(f"\ntape polish → ring-deform ({len(args.target)} targets)")
        return subprocess.run(cmd).returncode

    # No tape targets — just measure the betas fit for CSV / SMIS / OBJ.
    out_csv = out_prefix.with_name(out_prefix.name + "_measurements.csv")
    out_obj = out_prefix.with_name(out_prefix.name + "_fit_body.obj")
    out_smis = out_prefix.with_name(out_prefix.name + ".smis")
    cmd = [
        sys.executable, "-m", "tailor_twin.measure.cli", str(out_npz),
        "--num-betas", str(num_betas), "--gender", gender,
        "--model-folder", args.model_folder,
        "--save-csv", str(out_csv),
        "--save-obj", str(out_obj),
        "--save-smis", str(out_smis),
    ]
    r = subprocess.run(cmd)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
