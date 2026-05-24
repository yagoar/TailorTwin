"""CLI: fit SMPL-X shape to two photos via Sapiens2 pointmaps.

Same goal as ``silhouette-fit`` — front + side photo + height → an
SMPL-X body — but the torso width/depth curves come from Sapiens2's
**metric pointmap** instead of a 2-D outline.

Why this drops the tape polish
-------------------------------
``silhouette-fit`` derives width and depth from a binary mask scaled by
one global pixel→cm factor; both inherit that factor's error, so the
absolute girths can be off enough to need a ``ring-deform`` tape polish
(which then distorts the outline). Sapiens2's pointmap gives a metric
(x, y, z) per pixel: width and depth are each measured directly in
metres and rescaled once by the known height. On the project's captures
this matches tape chest/​hip within ~1 cm with no tape input.

Pipeline
--------
1. Sapiens2 ``vis_pointmap`` + ``vis_seg`` on both photos (cached).
2. :func:`~tailor_twin.fit.pointmap.pointmap_profile` → metric torso
   width (front) and depth (side) curves, arms removed via part-seg.
3. Leg width/depth from the part-seg silhouette (``extract_profile``).
4. :func:`~tailor_twin.fit.silhouette_betas.fit_betas_to_silhouette`
   solves the SMPL-X betas — unchanged solver, metric targets.
5. Mesh rebuilt, height-scaled, measured → npz / CSV / OBJ / SMIS.

Example::

    tailor-twin pointmap-fit data/captures/me_photos/front.jpg \\
        data/captures/me_photos/side.jpg --height 160 \\
        --base-fit data/results/me_shapy_smplx_fit.npz \\
        --out-prefix data/results/me_pointmap
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("front", type=Path, help="Front-view RGB photo.")
    p.add_argument("side", type=Path, help="Side-view (90°) RGB photo.")
    p.add_argument("--height", type=float, required=True,
                   help="Subject height in cm (absolute scale reference).")
    p.add_argument("--base-fit", type=Path, required=True,
                   help="Existing fit npz — seeds betas + supplies gender.")
    p.add_argument("--out-prefix", type=Path, required=True)
    p.add_argument("--model-folder", default="data/body_models")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="Where Sapiens artifacts are cached "
                        "(default: <out-prefix>_sapiens).")
    p.add_argument("--model-size", default="0.4b",
                   choices=["0.4b", "0.8b", "1b", "5b"],
                   help="Sapiens2 pointmap model size (needs the matching "
                        "checkpoint).")
    p.add_argument("--seg-size", default="0.4b",
                   choices=["0.4b", "0.8b", "1b", "5b"],
                   help="Sapiens2 part-seg model size — used only for arm "
                        "masking, so the cheap 0.4b is enough.")
    p.add_argument("--device", default=None,
                   help="Inference device (default: auto — mps/cuda/cpu).")
    p.add_argument("--n-active", type=int, default=10,
                   help="Number of leading SMPL-X betas to optimize. Kept at "
                        "10 — the dominant CAESAR shape modes. Fitting more "
                        "lets high-order modes drift to extreme values "
                        "(8-sigma) that warp unconstrained regions like the "
                        "head; the torso slices do not pin them down.")
    p.add_argument("--n-slices", type=int, default=24,
                   help="Torso control slices between --y-lo and --y-hi.")
    p.add_argument("--y-lo", type=float, default=0.50,
                   help="Lowest torso slice, fraction of body height.")
    p.add_argument("--y-hi", type=float, default=0.82,
                   help="Highest torso slice, fraction of body height.")
    p.add_argument("--max-iters", type=int, default=40,
                   help="Gauss-Newton iterations. 18 stops before the "
                        "betas settle — 40 lets the hip slices converge.")
    p.add_argument("--girth-weight", type=float, default=1.0,
                   help="Weight of the per-slice girth (perimeter) "
                        "residual. Width/depth alone leave the section "
                        "free to bow out fuller — this holds the mesh "
                        "ring to the photo-measured circumference. "
                        "0 disables it.")
    p.add_argument("--no-legs", action="store_true",
                   help="Skip the leg fit — torso betas only.")
    p.add_argument("--fit-arms", action="store_true",
                   help="Also fit arm width from the front pointmap "
                        "(metric, PCA axis-aligned). Off by default — arm "
                        "betas compete with torso/leg betas.")
    p.add_argument("--a-pose-shoulder-deg", type=float, default=30.0,
                   help="Canonical A-pose shoulder angle (0 = T-pose).")
    args = p.parse_args(argv)

    for ph in (args.front, args.side, args.base_fit):
        if not ph.exists():
            raise SystemExit(f"missing input: {ph}")

    work_dir = args.work_dir or args.out_prefix.with_name(
        args.out_prefix.name + "_sapiens")

    # --- 1. Sapiens2 pointmap + part-seg -------------------------------
    from .pointmap import (load_pose_keypoints, pointmap_profile, run_pose,
                           run_sapiens)
    art = run_sapiens({"front": args.front, "side": args.side}, work_dir,
                      model_size=args.model_size, seg_size=args.seg_size,
                      device=args.device)

    # --- 1b. pose keypoints — arm masking ------------------------------
    # The part-seg only catches the forearm/hand; pose keypoints let
    # pointmap_profile cut the upper arm too (critical when the arms are
    # spread). Degrades to seg-only arm removal if pose is unavailable.
    pose_kp: dict = {}
    try:
        in_dir = art["front"]["photo"].parent
        pose_out = work_dir / "pose"
        cached = sorted(pose_out.glob("*_predictions.json")) \
            if pose_out.exists() else []
        js = cached[-1] if cached else run_pose(
            in_dir, pose_out, device=args.device)
        pose_kp = load_pose_keypoints(js)
        print(f"pose: arm-mask keypoints for {sorted(pose_kp)}")
    except SystemExit as e:
        print(f"pose unavailable ({e}) — arm masking is seg-only")

    def _kp(name):
        v = pose_kp.get(name)
        return (v[0], v[1]) if v else (None, None)

    # --- 2. metric torso profiles --------------------------------------
    # Pose arm-masking is applied to the front only — in the side view
    # the arm overlaps the torso, so a tube would cut real body; the
    # side keeps seg-only arm removal.
    print("building metric torso profiles from pointmaps …")
    fk, fs = _kp("front")
    front_prof = pointmap_profile(
        art["front"]["ply"], art["front"]["seg"], art["front"]["photo"],
        args.height, view="front", pose_kp=fk, pose_sc=fs)
    side_prof = pointmap_profile(
        art["side"]["ply"], art["side"]["seg"], art["side"]["photo"],
        args.height, view="side")
    fv = int(front_prof.valid.sum())
    sv = int(side_prof.valid.sum())
    print(f"  front: {fv} valid rows, {front_prof.height_px:.0f} px tall")
    print(f"  side : {sv} valid rows, {side_prof.height_px:.0f} px tall")

    # --- 3a. arm width profile (opt-in) --------------------------------
    arm_prof = None
    if args.fit_arms:
        from .pointmap import pointmap_arm_profile
        arm_prof = pointmap_arm_profile(
            art["front"]["ply"], art["front"]["seg"], args.height)
        ok = int(np.sum(np.isfinite(arm_prof)))
        print(f"  arm: {ok}/{len(arm_prof)} valid bins (front pointmap)")

    # --- 3. leg profiles from the part-seg silhouette ------------------
    front_leg_prof = side_leg_prof = None
    if not args.no_legs:
        from .silhouette import extract_profile, load_silhouette
        front_mask, _ = load_silhouette(str(art["front"]["seg"]))
        side_mask, _ = load_silhouette(str(art["side"]["seg"]))
        front_leg_prof = extract_profile(front_mask, args.height,
                                         view="front", pick="widest")
        side_leg_prof = extract_profile(side_mask, args.height,
                                        view="side", pick="widest")

    # --- 4. betas fit ---------------------------------------------------
    fit = np.load(args.base_fit)
    from ..fit.fit import fit_gender
    gender = fit_gender(fit)
    base_betas = fit["betas"].astype(np.float64)
    num_betas = base_betas.shape[0]

    from .silhouette_betas import fit_betas_to_silhouette
    result = fit_betas_to_silhouette(
        base_betas, gender, front_prof, side_prof,
        height_cm=args.height,
        front_leg_prof=front_leg_prof, side_leg_prof=side_leg_prof,
        arm_prof=arm_prof,
        model_folder=args.model_folder,
        a_pose_shoulder_deg=args.a_pose_shoulder_deg,
        n_active=args.n_active, n_slices=args.n_slices,
        y_lo=args.y_lo, y_hi=args.y_hi, max_iters=args.max_iters,
        girth_w=args.girth_weight)

    print("\ntorso slices — pointmap target vs fitted mesh (cm):")
    print(f"  {'Y':>5}  {'width tgt/got':>16}  {'depth tgt/got':>16}")
    step = max(args.n_slices // 8, 1)
    for i in range(0, args.n_slices, step):
        print(f"  {result.y_norm[i]:5.2f}  "
              f"{result.width_target[i]:7.1f}/{result.width_after[i]:7.1f}  "
              f"{result.depth_target[i]:7.1f}/{result.depth_after[i]:7.1f}")

    if result.leg_y_norm is not None:
        print("\nleg slices — silhouette target vs fitted mesh (cm):")
        nleg = len(result.leg_y_norm)
        for i in range(0, nleg, max(nleg // 6, 1)):
            print(f"  {result.leg_y_norm[i]:5.2f}  "
                  f"{result.leg_width_target[i]:7.1f}/"
                  f"{result.leg_width_after[i]:7.1f}  "
                  f"{result.leg_depth_target[i]:7.1f}/"
                  f"{result.leg_depth_after[i]:7.1f}")

    if result.arm_width_target is not None:
        print("\narm width — pointmap target vs fitted mesh (cm, "
              "shoulder→wrist):")
        for i in range(len(result.arm_width_target)):
            t = result.arm_width_target[i]
            g = result.arm_width_after[i]
            ts = f"{t:6.1f}" if np.isfinite(t) else "    --"
            gs = f"{g:6.1f}" if np.isfinite(g) else "    --"
            print(f"  bin {i}  tgt {ts}  got {gs}")

    # --- 5. rebuild mesh, height-scale, save ---------------------------
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

    y_min = float(verts[:, 1].min())
    cur_h_cm = (verts[:, 1].max() - y_min) * 100.0
    s_h = args.height / cur_h_cm
    verts[:, 1] = y_min + (verts[:, 1] - y_min) * s_h
    print(f"\nheight: {cur_h_cm:.1f} → ×{s_h:.4f} → {args.height} cm")

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
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
