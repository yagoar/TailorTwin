"""Hybrid Anny fit: Multi-HMR regresses pose from photo, anny_solve adds
tape-exact shape on top.

Multi-HMR Anny gets the body POSE right (163 bone rotations from each
photo) but its base ``local_changes=False`` Anny model can't represent
bust / waist / hip-circumference variation — those rely on the
``measure-*-circ-incr`` blendshapes that Multi-HMR's regression head
doesn't predict.

Pipeline:
  1. Load Multi-HMR per-view inference (front + side) — pre-computed
     by ``run_inference.py`` in the multi-hmr venv → ``.pt`` files.
  2. Build the FULL Anny model with ``local_changes="all"`` so the
     tape blendshapes exist.
  3. Override phenotype with user-known values (gender, age years,
     height cm, weight kg) — Multi-HMR's regressed shape is noisy
     (age 43 vs real 30, weight low).
  4. Solve ``measure-*-circ-incr`` blendshapes from tape (anny_solve
     1D root-find per girth).
  5. Use Multi-HMR's bone rotations as the body pose (per view, or
     averaged / front-only — TBD).
  6. Render mesh + save fit npz.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import anny

# Reuse the solver helpers
import sys
sys.path.insert(0, str(Path(__file__).parent))
from anny_solve import AnnyFitter, TAPE_PLAN, bisect


torch.set_default_dtype(torch.float32)


def load_multihmr(pt_path: Path) -> dict:
    """Return the per-person dict saved by run_inference.py."""
    return torch.load(pt_path, weights_only=False, map_location="cpu")


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
                    help="Sapiens side_seg — used for landmark Y fractions.")
    ap.add_argument("--pose-source", choices=("front", "side", "avg"),
                    default="front",
                    help="Which Multi-HMR view supplies the body pose. "
                         "Front view = arms/hands pose; side view = "
                         "shoulder rotation. Avg = mean rotvec.")
    args = ap.parse_args(argv)

    # Photo landmarks JSON
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

    # Multi-HMR predictions per view
    front_h = load_multihmr(args.multihmr_front)
    side_h = load_multihmr(args.multihmr_side)
    print(f"multihmr front: {len(front_h['rotvec'])} bones, "
          f"shape {front_h['shape'].shape}")

    # Build Anny fitter (full local_changes). Force float64 so Multi-HMR's
    # rotvec (also float64) doesn't trip the model's internal dtype checks.
    f = AnnyFitter(gender=args.gender, age_years=args.age)
    f.bm = f.bm.double()
    # Re-cast existing A-pose dict and phenotype tensors.
    f.a_pose = {k: v.double() for k, v in f.a_pose.items()}
    print(f"Anny model: gender={args.gender} age={args.age}y "
          f"→ phenotype age={f.age:.2f}")

    # Step 1: pull body pose from Multi-HMR. ``rotvec`` is (163, 3)
    # per-bone axis-angle. Convert to (163, 4, 4) homogeneous matrices
    # and dispatch as ``pose_parameters`` named-bone dict.
    import roma
    if args.pose_source == "avg":
        pose_rotvec = (front_h["rotvec"] + side_h["rotvec"]) / 2.0
    elif args.pose_source == "side":
        pose_rotvec = side_h["rotvec"]
    else:
        pose_rotvec = front_h["rotvec"]
    # Anny uses float64 internally — convert rotvec.
    pose_rotvec = pose_rotvec.double()
    rotmat = roma.rotvec_to_rotmat(pose_rotvec)        # (163, 3, 3) f64
    homo = torch.eye(4, dtype=torch.float64).unsqueeze(0).repeat(
        rotmat.shape[0], 1, 1)
    homo[:, :3, :3] = rotmat
    bone_labels = f.bm.bone_labels
    # Multi-HMR's root rotation is the body's global orientation in
    # camera space — replacing Anny's canonical upright root with it
    # tips / collapses the model. Skip bone 0 (root) and any other
    # global-frame bones — keep only joint rotations (arms, head,
    # legs) which ARE rest-relative.
    pose_dict = {bone_labels[i]: homo[i].unsqueeze(0)
                 for i in range(1, len(bone_labels))}
    # Overwrite the A-pose dict on the fitter — keep canonical foot/hand
    # poses if missing from Multi-HMR (already eye).
    f.a_pose = pose_dict
    print(f"pose source: {args.pose_source}  "
          f"({sum(1 for v in pose_dict.values() if not torch.allclose(v[0], torch.eye(4, dtype=v.dtype)))} "
          f"non-identity bones)")

    # Step 2: phenotype height + weight
    height_cm = float(measurements.get("height", 160.0))
    def h_of(coef):
        f.pheno["height"] = coef; return f.anth_height_cm()
    f.pheno["height"] = bisect(h_of, 0.0, 1.0, target=height_cm, tol=0.1)
    print(f"phenotype.height = {f.pheno['height']:.3f}  "
          f"→ {f.anth_height_cm():.1f}cm (target {height_cm})")

    weight_kg = float(measurements.get("weight_kg", 57.0))
    def w_of(coef):
        f.pheno["weight"] = coef; return f.anth_mass_kg()
    f.pheno["weight"] = bisect(w_of, 0.0, 1.0, target=weight_kg, tol=0.2)
    f.pheno["height"] = bisect(h_of, 0.0, 1.0, target=height_cm, tol=0.1)
    print(f"phenotype.weight = {f.pheno['weight']:.3f}  "
          f"→ {f.anth_mass_kg():.1f}kg (target {weight_kg})")

    # Step 3: tape girth blendshapes
    print("\ntape fits (with Multi-HMR pose):")
    print(f"  {'name':10}  {'target':>7}  {'baseline':>8}  {'coef':>6}  {'final':>7}")
    for lm_key, blendshape, region, default_y in TAPE_PLAN:
        target_key = "knee_circ" if lm_key == "knee" else lm_key
        target_cm = measurements.get(target_key)
        if target_cm is None or blendshape is None:
            continue
        target_cm = float(target_cm)
        y_frac_from_top = lm_y_frac.get(lm_key, default_y)
        y_frac_from_feet = 1.0 - y_frac_from_top
        if lm_key == "waist":
            def g(coef, key=blendshape):
                f.lc[key] = coef; return f.anth_waist_cm()
            baseline = f.anth_waist_cm()
        else:
            def g(coef, key=blendshape, yf=y_frac_from_feet, reg=region):
                f.lc[key] = coef
                return f.girth_at_y(yf, region=reg)
            baseline = f.girth_at_y(y_frac_from_feet, region=region)
        coef = bisect(g, -1.5, 1.5, target=target_cm, tol=0.1)
        f.lc[blendshape] = coef
        final = (f.anth_waist_cm() if lm_key == "waist"
                  else f.girth_at_y(y_frac_from_feet, region=region))
        print(f"  {lm_key:10}  {target_cm:>7.1f}  {baseline:>8.1f}  "
              f"{coef:>+6.2f}  {final:>7.1f}")

    print(f"\nfinal: H={f.anth_height_cm():.1f}cm  W={f.anth_mass_kg():.1f}kg")

    # Save fit + render
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_npz = args.out_prefix.with_name(args.out_prefix.name + "_hybrid_fit.npz")
    verts = f.verts()
    np.savez(out_npz,
             phenotype={k: float(v) for k, v in f.pheno.items()},
             local_changes={k: float(v) for k, v in f.lc.items()},
             pose_source=args.pose_source,
             vertices=verts.astype(np.float32),
             faces=f.bm.get_triangular_faces().cpu().numpy().astype(np.int32))
    print(f"wrote {out_npz}")
    # OBJ
    obj = args.out_prefix.with_name(args.out_prefix.name + "_hybrid.obj")
    faces = f.bm.get_triangular_faces().cpu().numpy()
    with open(obj, "w") as fh:
        for x, y, z in verts: fh.write(f"v {x} {y} {z}\n")
        for a, b, c in faces: fh.write(f"f {a+1} {b+1} {c+1}\n")
    print(f"wrote {obj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
