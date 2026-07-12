"""CLI: deform a fit's mesh so PlanarGirth circumferences hit tape exactly.

Unlike ``refine-tape`` (which solves SMPL-X betas and is bounded by the
shape space), this edits mesh geometry per-ring. Every targeted girth
lands on its target; no shape-space coupling.

Only PlanarGirth-style codes are deformable (horizontal-slice
circumferences): G04 bust, G05 lowbust, G07 waist, G08 highhip,
G09 hip. The Y level of each comes from the fit's LandmarkSet.

Example::

    tailor-twin ring-deform data/results/me_smplx_fit.npz \\
        --out-prefix data/results/me_ringdeform \\
        --target G04=88 --target G07=70 --target G09=99
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


# seamly code -> the LandmarkSet key whose Y defines the slice plane.
# Mirrors the PlanarGirth(...) landmark args in measure/seamly_catalog.py.
# G03 highbust is a TapeLoop (not PlanarGirth), but its tape path runs
# at the underarm Y level — radially scaling that Y-band still shortens
# the tape path, so it's driven the same extractor-feedback way.
_CODE_TO_LANDMARK = {
    "G03": "underarm_left",
    "G04": "bust_level",
    "G05": "lowbust_level",
    "G07": "waist_string",
    "G08": "high_hip_level",
    "G09": "hip_level",
}

# Leg-girth codes: each slices a single lower-limb at a per-leg landmark Y
# and is deformed PER LEG (each leg scaled about its own centroid). The
# torso ring deform can't do these — it scales about one shared torso
# centroid, which would push two separate legs together/apart instead of
# scaling each girth. Mirrors the PlanarGirth landmarks in seamly_catalog.
_LEG_CODE_TO_LANDMARK = {
    "M03": "thigh_at_crotch_left",
    "M05": "mid_knee_level",
    "M07": "calf_widest_left",
    "M09": "ankle_bone_lateral_left",
}


def _parse_target(spec: str) -> tuple[str, float]:
    if "=" not in spec:
        raise SystemExit(f"--target expects CODE=value_cm, got {spec!r}")
    code, val = spec.split("=", 1)
    return code.strip(), float(val)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("fit_npz", type=Path)
    p.add_argument("--out-prefix", type=Path, required=True)
    p.add_argument("--target", action="append", required=True,
                   help="CODE=value_cm. Repeatable. PlanarGirth codes only "
                        "(G04/G05/G07/G08/G09).")
    p.add_argument("--model-folder", default="data/body_models")
    p.add_argument("--num-betas", type=int, default=300)
    p.add_argument("--band-m", type=float, default=0.06,
                   help="Half-height of the deformed Y-band (m).")
    p.add_argument("--passes", type=int, default=4)
    p.add_argument("--waist-y", type=float, default=None,
                   help="Override waist slice Y (m). Else from landmarks.")
    p.add_argument("--waist-height-cm", type=float, default=None,
                   help="Tape-measured waist height (floor → natural waist, "
                        "vertical, cm). Frame-robust alternative to "
                        "--waist-y: the G07 slice Y is re-derived as "
                        "mesh min Y + height on the re-posed mesh each "
                        "pass, so it survives the canonical re-centre and "
                        "the A01 height scale. --waist-y wins if both set.")
    p.add_argument("--a-pose-shoulder-deg", type=float, default=30.0,
                   help="Re-pose the fit to canonical A-pose before "
                        "deforming (0 = T-pose). The chamfer-fit mesh "
                        "carries the scan-time pose; this discards it so "
                        "the deformed OBJ is in a clean garment pose.")
    args = p.parse_args(argv)

    targets_cm = dict(_parse_target(s) for s in args.target)
    # A01 height is handled by a uniform Y-scale, not a ring.
    height_target = targets_cm.pop("A01", None)
    # Split leg-girth targets out — they need a per-leg deform, not the
    # shared-centroid torso profile.
    leg_targets = {c: targets_cm.pop(c) for c in list(targets_cm)
                   if c in _LEG_CODE_TO_LANDMARK}
    bad = [c for c in targets_cm if c not in _CODE_TO_LANDMARK]
    if bad:
        raise SystemExit(
            f"codes {bad} not deformable; supported: A01 (height) + "
            f"{sorted(_CODE_TO_LANDMARK)} + legs {sorted(_LEG_CODE_TO_LANDMARK)}")

    fit = np.load(args.fit_npz)

    from ..fit.fit import fit_gender
    from ..measure.landmarks import build_landmark_set
    import smplx
    import torch

    gender = fit_gender(fit)
    bm = smplx.create(model_path=args.model_folder, model_type="smplx",
                      gender=gender, num_betas=args.num_betas,
                      use_pca=False, flat_hand_mean=True, batch_size=1)
    faces = np.asarray(bm.faces, dtype=np.int32)

    # Re-pose to canonical A-pose. The chamfer-fit mesh carries the
    # scan-time pose (twisted torso, asymmetric arms); ring deformation
    # only edits girths, so without this the OBJ stays in that pose.
    # Regenerate vertices from the fit's betas + a canonical pose.
    from .refine_to_tape import _build_a_pose
    betas = fit["betas"].astype(np.float32)
    canon_pose = _build_a_pose(args.a_pose_shoulder_deg).astype(np.float32)
    with torch.no_grad():
        out = bm(
            betas=torch.from_numpy(betas[None, :]),
            body_pose=torch.from_numpy(canon_pose.reshape(1, -1)),
            global_orient=torch.zeros(1, 3),
            transl=torch.zeros(1, 3),
            return_full_pose=False,
        )
    verts = out.vertices[0].cpu().numpy().astype(np.float64)
    joints = out.joints[0].cpu().numpy().astype(np.float32)
    disp = fit["displacement"] if "displacement" in fit.files else None
    if disp is not None and disp.shape == verts.shape:
        verts = verts + disp

    landmarks = build_landmark_set(
        verts.astype(np.float32), joints=joints, faces=faces,
        gender=gender)

    from .ring_deform import RingTarget, apply_scale_profile
    from ..measure.regions import region_vertex_mask
    from ..measure.seamly_extractor import extract_catalog

    # Per-code vertex region — the set of verts a ring may move. Matches
    # the `regions=` arg of each code's PlanarGirth in seamly_catalog.py.
    # G09 hip slices through the upper legs, so its region includes them.
    _CODE_REGIONS = {
        "G03": ("torso",),
        "G04": ("torso",),
        "G05": ("torso",),
        "G07": ("torso",),
        "G08": ("torso",),
        "G09": ("torso", "left_leg", "right_leg"),
    }
    region_masks = {
        code: region_vertex_mask(regs, model_folder=args.model_folder,
                                  gender=gender)
        for code, regs in _CODE_REGIONS.items()
    }
    # Union mask for the continuous profile — torso + legs (legs only
    # carry a non-unity scale near the hip control point; the profile's
    # out-of-range ramp returns them to 1.0 well above the knee).
    deform_mask = region_vertex_mask(
        ("torso", "left_leg", "right_leg"),
        model_folder=args.model_folder, gender=gender)

    def _waist_override_y(v: np.ndarray) -> float | None:
        """G07 slice-Y override, re-derived against the mesh as it stands."""
        if args.waist_y is not None:
            return float(args.waist_y)
        if args.waist_height_cm is not None:
            from ..measure.landmarks import waist_y_from_height
            return waist_y_from_height(v, args.waist_height_cm)
        return None

    ring_targets: list[RingTarget] = []
    for code, target_cm in targets_cm.items():
        lm_name = _CODE_TO_LANDMARK[code]
        wy = _waist_override_y(verts) if code == "G07" else None
        if wy is not None:
            y = wy
        else:
            y = float(landmarks[lm_name][1])
        ring_targets.append(RingTarget(
            code=code, y_level=y, target_cm=target_cm, band_m=args.band_m,
            region_mask=region_masks[code]))
        print(f"{code}: slice Y={y:.4f} m, target {target_cm} cm")

    deformed = verts.copy()

    # Height first — uniform Y-scale about the feet so A01 hits target.
    # Done before the rings because it shifts every slice Y; the ring
    # loop then re-reads landmark-relative circumferences on the scaled
    # mesh. (Y levels were captured pre-scale, so rescale them too.)
    if height_target is not None:
        cat0 = extract_catalog(
            deformed.astype(np.float32), faces, joints=joints, gender=gender)
        cur_h = cat0.values.get("A01")
        if cur_h and cur_h > 0:
            s = height_target / cur_h
            y_min = deformed[:, 1].min()
            deformed[:, 1] = y_min + (deformed[:, 1] - y_min) * s
            for t in ring_targets:
                t.y_level = y_min + (t.y_level - y_min) * s
            print(f"A01 height: {cur_h:.2f} → ×{s:.4f} → {height_target} cm")

    # Extractor-driven deformation loop. The real seamly extractor is
    # the single source of truth for the current circumference; the
    # geometric scale per ring is target / extractor_current. Iterating
    # re-converges because each ring's cosine falloff perturbs neighbours.
    for p in range(args.passes):
        # Rebuild landmarks each pass — slice Y levels (high_hip_level
        # etc.) drift as the mesh deforms. Stale Y scales the wrong band.
        lm = build_landmark_set(
            deformed.astype(np.float32), joints=joints, faces=faces,
            gender=gender)
        cat = extract_catalog(
            deformed.astype(np.float32), faces, joints=joints,
            gender=gender, landmarks=lm)

        ys: list[float] = []
        scales: list[float] = []
        max_resid = 0.0
        for t in ring_targets:
            current = cat.values.get(t.code)
            if current is None or not (current > 0):
                print(f"  pass {p+1} {t.code}: extractor gave no value, skip")
                continue
            lm_name = _CODE_TO_LANDMARK[t.code]
            wy = _waist_override_y(deformed) if t.code == "G07" else None
            if wy is not None:
                t.y_level = wy
            else:
                t.y_level = float(lm[lm_name][1])
            scale = t.target_cm / current
            ys.append(t.y_level)
            scales.append(scale)
            max_resid = max(max_resid, abs(current - t.target_cm))
            print(f"  pass {p+1} {t.code}: {current:.2f} cm "
                  f"→ scale ×{scale:.4f} (target {t.target_cm})")
        if max_resid <= 0.3:
            print(f"ring deform converged pass {p+1} "
                  f"(max residual {max_resid:.2f} cm)")
            break
        if len(ys) >= 2:
            # One continuous profile pass — smooth, no surface banding.
            deformed = apply_scale_profile(
                deformed, np.array(ys), np.array(scales),
                region_mask=deform_mask)
        elif len(ys) == 1:
            from .ring_deform import apply_radial_scale
            deformed = apply_radial_scale(
                deformed, ys[0], ring_targets[0].band_m,
                scales[0], deform_mask)

    # ---- Per-leg girth deform (M03 thigh / M05 knee / M07 calf / M09
    # ankle). Each leg is scaled about its OWN centroid (single-region
    # radial scale) so the two separate limbs aren't pushed together as a
    # shared-centroid torso ring would do. The scale per pass is driven by
    # the real seamly EXTRACTOR value (same source of truth as the torso
    # loop), not deform_ring's internal slice measure — the two disagree at
    # the knee/ankle where the cross-section is complex, which left those
    # codes short. The body is symmetrised, so the same scale applies to
    # both legs.
    if leg_targets:
        from .ring_deform import apply_radial_scale
        leg_masks = {
            side: region_vertex_mask((side,), model_folder=args.model_folder,
                                     gender=gender)
            for side in ("left_leg", "right_leg")
        }
        for code, target_cm in leg_targets.items():
            lm_name = _LEG_CODE_TO_LANDMARK[code]
            for p in range(args.passes):
                lm = build_landmark_set(
                    deformed.astype(np.float32), joints=joints, faces=faces,
                    gender=gender)
                cat = extract_catalog(
                    deformed.astype(np.float32), faces, joints=joints,
                    gender=gender, landmarks=lm)
                current = cat.values.get(code)
                if current is None or not (current > 0):
                    print(f"  leg {code}: extractor gave no value, skip")
                    break
                resid = abs(current - target_cm)
                print(f"  pass {p+1} {code}: {current:.2f} cm "
                      f"(target {target_cm}, resid {resid:.2f})")
                if resid <= 0.3:
                    print(f"leg {code} converged pass {p+1}")
                    break
                scale = target_cm / current
                y = float(lm[lm_name][1])
                for side in ("left_leg", "right_leg"):
                    deformed = apply_radial_scale(
                        deformed, y, args.band_m, scale, leg_masks[side])

    # Save a fit npz mirroring the source with deformed vertices.
    # Pose fields are overwritten with the canonical A-pose used above
    # so the npz is internally consistent (verts ↔ pose) and the
    # bent-arm re-pose offsets from a known base.
    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_npz = out_prefix.with_name(out_prefix.name + "_smplx_fit.npz")
    payload = {k: fit[k] for k in fit.files}
    payload["smplx_vertices"] = deformed.astype(np.float32)
    payload["smplx_joints"] = joints
    payload["body_pose"] = canon_pose
    payload["global_orient"] = np.zeros((3,), dtype=np.float32)
    payload["transl"] = np.zeros((3,), dtype=np.float32)
    payload["z"] = np.array([])
    np.savez(out_npz, **payload)
    print(f"\nwrote {out_npz}")

    # Re-run the measure CLI for CSV / SMIS / OBJ.
    out_csv = out_prefix.with_name(out_prefix.name + "_measurements.csv")
    out_obj = out_prefix.with_name(out_prefix.name + "_fit_body.obj")
    out_smis = out_prefix.with_name(out_prefix.name + ".smis")
    cmd = [
        sys.executable, "-m", "tailor_twin.measure.cli", str(out_npz),
        "--num-betas", str(args.num_betas), "--gender", gender,
        "--model-folder", args.model_folder,
        "--save-csv", str(out_csv),
        "--save-obj", str(out_obj),
        "--save-smis", str(out_smis),
    ]
    r = subprocess.run(cmd)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
