"""Photo-driven Anny canonical-pose fit.

Architecture (replaces the broken anny_hybrid.py pose-before-tape mix):

  1. Multi-HMR provides photo-derived ``shape`` (11-D), of which the
     first 6 entries map to Anny's phenotype labels
     ``[gender, age, muscle, weight, height, proportions]``. Pose
     (163 bone rotvecs) is IGNORED here — pose is photo-frame and
     belongs in a separate overlay step, not in the shape pipeline.

  2. Phenotype seed = Multi-HMR ``shape[:6]`` averaged over front+side.

  3. Phenotype overrides (known biometrics): gender, age (years/55),
     height (bisected to anth_height_cm = user_cm), weight (bisected
     to anth_mass_kg = user_kg).

  4. ``muscle`` + ``proportions`` are KEPT from Multi-HMR — these
     two phenotype params carry the photo-derived body-shape info
     (slim vs broad torso, limb proportion) that the known
     biometrics don't constrain.

  5. Tape solver: same ``measure-*-circ-incr`` bisect as anny_solve,
     applied on top of the photo-seeded phenotype. All evaluations
     in canonical A-pose so slice geometry is well-defined.

Output: canonical mesh npz + obj. No pose mixing.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import anny

sys.path.insert(0, str(Path(__file__).parent))
from anny_solve import AnnyFitter, TAPE_PLAN, bisect


torch.set_default_dtype(torch.float32)


# Stack plan: when ``measure-*-circ-incr`` saturates at Anny's internal
# clamp of -1.5, drive the primary blendshape AND several secondary
# shape blendshapes proportionally from a single composite coefficient.
# Empirically this lets bicep/neck reach much smaller girths for thin
# subjects without distorting overall topology.
#
#   key:  blendshape lm_key (matches TAPE_PLAN[0])
#   val:  list of (anny_blendshape_name, weight_factor)
#         coef applied to a blendshape = composite_coef * weight_factor
STACK_PLAN = {
    # Monotone-thinning arm blendshapes only. Scale-horiz / scale-depth
    # have near-zero or non-monotone effect on bicep slice (4cm delta
    # tested) so they're excluded to keep bisect well-behaved.
    "bicep": [
        ("measure-upperarm-circ-incr", 1.0),
        ("l-upperarm-fat-incr",        1.0),
        ("r-upperarm-fat-incr",        1.0),
        ("l-upperarm-muscle-incr",     1.0),
        ("r-upperarm-muscle-incr",     1.0),
    ],
    "neck": [
        ("measure-neck-circ-incr",     1.0),
        ("neck-double-incr",           0.5),
    ],
}


def apply_stack(f, stack: list[tuple[str, float]], composite: float):
    """Apply composite coefficient to every blendshape in stack at its
    weight factor. Modifies f.lc in place."""
    for name, w in stack:
        f.lc[name] = composite * w


# Anatomical Y override (from-top fraction) for tapes whose user-marked
# JSON Y or TAPE_PLAN default lands at a degenerate / clipped slice
# position. Empirically chosen by scanning girth_at_y vs Y for default
# phenotype + photo seed.
#   bicep: JSON has no "bicep" entry → TAPE_PLAN default 0.25 lands at
#          shoulder cap (~34-45 cm). Mid-upper-arm sits ~0.33.
#   neck:  user's "neck" Y (0.16) is at base-of-neck where Anny's neck
#          slice region clips to 0. Real neck slice has support
#          0.12-0.16 — use 0.135 for adult Adam's-apple level.
TAPE_Y_OVERRIDE = {
    "bicep": 0.33,
    "neck":  0.135,
}


# Multi-HMR's body_model is built with all_phenotypes=True. The 11-D
# shape vector's column order = body_model.phenotype_labels (with
# 'race' filtered out). The first 6 columns correspond to the 6
# standard body phenotypes (same names as default Anny). The remaining
# 5 are extra (head/face params we don't use here).
MHR_PHENO_ORDER = ["gender", "age", "muscle", "weight", "height", "proportions"]


def load_mhr(path: Path) -> dict:
    return torch.load(path, weights_only=False, map_location="cpu")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--landmarks", type=Path, required=True)
    ap.add_argument("--multihmr-front", type=Path,
                    default=Path("data/results/multihmr_spike/our_out_front_p0.pt"))
    ap.add_argument("--multihmr-side", type=Path,
                    default=Path("data/results/multihmr_spike/our_out_side_p0.pt"))
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--gender", default="female")
    ap.add_argument("--age", type=float, default=30.0)
    ap.add_argument("--side-seg", type=Path, default=None,
                    help="Sapiens side_seg → landmark Y fractions.")
    ap.add_argument("--no-mhr-seed", action="store_true",
                    help="Skip Multi-HMR phenotype seeding (baseline = "
                         "default 0.5 phenotype). Use to A/B test the "
                         "benefit of the photo seed.")
    args = ap.parse_args(argv)

    # ── Landmarks JSON ──
    data = json.loads(args.landmarks.read_text())
    measurements = data.get("measurements") or {}
    lines = data.get("lines_y") or {}
    lm_y_frac: dict[str, float] = {}
    if args.side_seg and args.side_seg.exists():
        seg = np.load(args.side_seg)
        if seg.ndim == 3:
            seg = seg.argmax(0) if seg.shape[0] < seg.shape[-1] else seg.argmax(-1)
        ys = np.where(seg > 0)[0]
        s_top, s_bot = ys.min(), ys.max()
        for name, by in lines.items():
            if by and by.get("side") is not None:
                lm_y_frac[name] = (by["side"] - s_top) / (s_bot - s_top)
    print(f"landmark Y fracs: {lm_y_frac}")

    # ── Multi-HMR phenotype seed ──
    mhr_shape = None
    if not args.no_mhr_seed:
        front = load_mhr(args.multihmr_front)
        side = load_mhr(args.multihmr_side)
        mhr_shape = ((front["shape"] + side["shape"]) / 2.0).cpu().numpy()
        print(f"Multi-HMR shape[:6] (avg front+side): "
              f"{[f'{v:.3f}' for v in mhr_shape[:6]]}")
        print(f"  → {dict(zip(MHR_PHENO_ORDER, mhr_shape[:6].round(3)))}")

    # ── Build canonical AnnyFitter ──
    f = AnnyFitter(gender=args.gender, age_years=args.age)
    print(f"Anny model: gender={args.gender}, age={args.age}y "
          f"→ phenotype.age={f.age:.3f}")

    # Seed muscle + proportions from Multi-HMR (photo-derived body shape).
    # Keep gender/age locked to user-known values. Height/weight will be
    # re-bisected below from known cm/kg.
    if mhr_shape is not None:
        f.pheno["muscle"] = float(mhr_shape[MHR_PHENO_ORDER.index("muscle")])
        f.pheno["proportions"] = float(mhr_shape[MHR_PHENO_ORDER.index("proportions")])
        # Use MHR height/weight only as initial bisect bracket starting
        # point; the bisect below will refine to nail exact cm/kg.
        f.pheno["height"] = float(mhr_shape[MHR_PHENO_ORDER.index("height")])
        f.pheno["weight"] = float(mhr_shape[MHR_PHENO_ORDER.index("weight")])
        print(f"seeded muscle={f.pheno['muscle']:.3f}  "
              f"proportions={f.pheno['proportions']:.3f}  "
              f"(photo-derived, kept fixed through tape solve)")

    # ── Lock height ──
    height_cm = float(measurements.get("height", 160.0))
    def h_of(coef):
        f.pheno["height"] = coef; return f.anth_height_cm()
    f.pheno["height"] = bisect(h_of, 0.0, 1.0, target=height_cm, tol=0.1)
    print(f"phenotype.height = {f.pheno['height']:.3f}  "
          f"→ {f.anth_height_cm():.1f}cm  (target {height_cm})")

    # ── Lock weight ──
    weight_kg = float(measurements.get("weight_kg", 57.0))
    def w_of(coef):
        f.pheno["weight"] = coef; return f.anth_mass_kg()
    f.pheno["weight"] = bisect(w_of, 0.0, 1.0, target=weight_kg, tol=0.2)
    # Re-pin height (weight changes it).
    f.pheno["height"] = bisect(h_of, 0.0, 1.0, target=height_cm, tol=0.1)
    print(f"phenotype.weight = {f.pheno['weight']:.3f}  "
          f"→ {f.anth_mass_kg():.1f}kg  (target {weight_kg})")
    print(f"re-pin height:     {f.anth_height_cm():.1f}cm")

    # ── Pre-tape shape blendshapes (fixed coefs, not bisected). ──
    # ``breast-point-incr`` adds bust forward projection AND
    # circumference. Setting it positive before the tape solver lets
    # ``measure-bust-circ-incr`` go more negative to hit the same 88cm
    # target — net result: same circumference, real cup shape rather
    # than a flat chest. Without this the bust projects ~zero in the
    # side view since Anny default + negative bust-circ-incr both push
    # the chest flat.
    f.lc["breast-point-incr"] = 1.5
    f.lc["torso-scale-depth-incr"] = 0.3
    print("pre-tape shape: breast-point-incr=+1.5  torso-scale-depth-incr=+0.3")

    # ── Tape blendshapes ──
    print("\ntape girth fits (canonical pose, photo-seeded phenotype):")
    print(f"  {'name':10}  {'target':>7}  {'baseline':>8}  "
          f"{'coef':>6}  {'final':>7}")
    for lm_key, blendshape, region, default_y in TAPE_PLAN:
        target_key = "knee_circ" if lm_key == "knee" else lm_key
        target_cm = measurements.get(target_key)
        if target_cm is None or blendshape is None:
            continue
        target_cm = float(target_cm)
        y_frac_from_top = TAPE_Y_OVERRIDE.get(
            lm_key, lm_y_frac.get(lm_key, default_y))
        y_frac_from_feet = 1.0 - y_frac_from_top
        stack = STACK_PLAN.get(lm_key)
        measure_fn = ((lambda: f.anth_waist_cm()) if lm_key == "waist"
                       else (lambda yf=y_frac_from_feet, reg=region:
                             f.girth_at_y(yf, region=reg)))
        baseline = measure_fn()

        if stack:
            # Single composite coef drives the whole stack.
            def g(coef, st=stack):
                apply_stack(f, st, coef)
                return measure_fn()
        else:
            # Single-blendshape bisect (legacy path).
            def g(coef, key=blendshape):
                f.lc[key] = coef
                return measure_fn()
        coef = bisect(g, -1.5, 1.5, target=target_cm, tol=0.1)
        if stack:
            apply_stack(f, stack, coef)
        else:
            f.lc[blendshape] = coef
        final = measure_fn()
        tag = " stack" if stack else ""
        sat = " SAT" if abs(abs(coef) - 1.5) < 0.02 else ""
        print(f"  {lm_key:10}  {target_cm:>7.1f}  {baseline:>8.1f}  "
              f"{coef:>+6.2f}  {final:>7.1f}{tag}{sat}")

    # ── Re-pin height + weight after tape blendshapes ──
    # Tape ``measure-*-circ-incr`` shifts mesh mass (negative coefs
    # remove flesh), so anth_mass_kg drifts off target. Re-bisect
    # phenotype.weight / height to lock them.
    print("\nre-pin after tape:")
    f.pheno["weight"] = bisect(w_of, 0.0, 1.0, target=weight_kg, tol=0.2)
    f.pheno["height"] = bisect(h_of, 0.0, 1.0, target=height_cm, tol=0.1)
    print(f"  weight = {f.pheno['weight']:.3f}  → {f.anth_mass_kg():.1f}kg")
    print(f"  height = {f.pheno['height']:.3f}  → {f.anth_height_cm():.1f}cm")

    print(f"\nfinal: H={f.anth_height_cm():.1f}cm  "
          f"W={f.anth_mass_kg():.1f}kg")

    # ── Save canonical fit (tape solve T-pose for measurement). ──
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_npz = args.out_prefix.with_name(args.out_prefix.name + "_photofit.npz")
    verts = f.verts()
    faces_np = f.bm.get_triangular_faces().cpu().numpy()
    np.savez(out_npz,
             phenotype={k: float(v) for k, v in f.pheno.items()},
             local_changes={k: float(v) for k, v in f.lc.items()},
             mhr_shape=(mhr_shape if mhr_shape is not None
                        else np.zeros(0, dtype=np.float32)),
             gender=args.gender, age_years=args.age,
             vertices=verts.astype(np.float32),
             faces=faces_np.astype(np.int32))
    print(f"wrote {out_npz}")
    obj = args.out_prefix.with_name(args.out_prefix.name + "_photofit.obj")
    with open(obj, "w") as fh:
        for x, y, z in verts: fh.write(f"v {x} {y} {z}\n")
        for a, b, c in faces_np: fh.write(f"f {a+1} {b+1} {c+1}\n")
    print(f"wrote {obj}")

    # ── Save per-view photo-pose meshes (RENDER ONLY). ──
    # Same phenotype + lc, photo-derived bone rotvecs from each view.
    # Tape numbers are NOT recomputed in these poses — slice geometry
    # only makes sense in canonical pose, but the photo-matched pose
    # gives a fair body-shape comparison against the photo silhouette.
    if not args.no_mhr_seed:
        import roma
        labels = f.bm.bone_labels
        for view, mhr_path in (("front", args.multihmr_front),
                                ("side",  args.multihmr_side)):
            mhr = load_mhr(mhr_path)
            rotvec = mhr["rotvec"].cpu().numpy()
            rotmat = roma.rotvec_to_rotmat(torch.from_numpy(rotvec)).numpy()
            homo = np.tile(np.eye(4, dtype=np.float32),
                           (rotmat.shape[0], 1, 1))
            homo[:, :3, :3] = rotmat
            pose_dict = {labels[i]: torch.from_numpy(homo[i])[None]
                         for i in range(1, len(labels))}
            f.a_pose = pose_dict
            render_verts = f.verts()
            r_npz = args.out_prefix.with_name(
                args.out_prefix.name + f"_render_{view}.npz")
            r_obj = args.out_prefix.with_name(
                args.out_prefix.name + f"_render_{view}.obj")
            np.savez(r_npz,
                     vertices=render_verts.astype(np.float32),
                     faces=faces_np.astype(np.int32))
            with open(r_obj, "w") as fh:
                for x, y, z in render_verts: fh.write(f"v {x} {y} {z}\n")
                for a, b, c in faces_np: fh.write(f"f {a+1} {b+1} {c+1}\n")
            print(f"wrote {r_npz}  ({view} Multi-HMR pose, RENDER ONLY)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
