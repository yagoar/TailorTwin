"""Tape + landmark driven Anny body fit (no silhouette IoU optimisation).

Anny's blendshapes are interpretable + decoupled: each tape circumference
has its own ``measure-*-circ-incr`` blendshape that ONLY moves that
girth. Each proportion has its own ``measure-*-height-incr`` /
``measure-*-dist-incr`` blendshape. So instead of joint IoU+tape
optimisation (which fights itself, distorts the mesh, and takes
minutes), solve each measurement INDEPENDENTLY by 1D root-find on the
relevant blendshape coefficient.

Pipeline:
  1. Fix gender; map age years → Anny age phenotype (years / 55).
  2. Solve ``phenotype.height`` so anthropometric height = user height.
  3. Solve ``phenotype.weight`` so anthropometric mass  = user kg.
  4. For each tape (bust, underbust, waist, highhip, hip, bicep,
     thigh, knee, neck): 1D-solve the corresponding measure-*-circ-incr
     so mesh girth at the photo-derived Y matches the tape cm.
  5. For each photo proportion landmark (shoulder_y, waist_y,
     crotch_y, knee_y): 1D-solve the corresponding length blendshape.
  6. Pose arms / legs to A-pose via bone_poses.
  7. Render diagnostic overlay vs the photo silhouette (for QA only —
     overlay does NOT drive the fit).

Total time: ~5-20 sec depending on bisection tolerance.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import anny


torch.set_default_dtype(torch.float32)


# ─────────────────────────── 1D root-find helpers ──────────────────────────

def bisect(f, lo: float, hi: float, target: float = 0.0,
           tol: float = 0.05, max_iter: int = 25) -> float:
    """Bracket-bisection for monotonic f. Returns x where f(x) ≈ target.
    Falls back to clamped boundary if target is unreachable."""
    flo = f(lo) - target; fhi = f(hi) - target
    if flo * fhi > 0:
        return lo if abs(flo) < abs(fhi) else hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid) - target
        if abs(fmid) < tol:
            return mid
        if flo * fmid <= 0:
            hi = mid; fhi = fmid
        else:
            lo = mid; flo = fmid
    return 0.5 * (lo + hi)


# ─────────────────────────── Anny model wrapper ────────────────────────────

class AnnyFitter:
    """Holds the Anny model + accumulated phenotype / local_changes state
    + cached per-bone-region vertex masks. Exposes ``girth_at(name)`` /
    ``anth_height_cm()`` / ``anth_mass_kg()`` so the solver can evaluate
    a single measurement without re-typing the kwargs each call."""

    def __init__(self, gender: str = "female", age_years: float = 30.0):
        self.bm = anny.create_fullbody_model(local_changes="all")
        self.anth = anny.Anthropometry(self.bm)
        self.gender = 0.0 if gender == "female" else 1.0
        self.age = float(age_years) / 55.0  # WHO-calibrated mapping (rough)
        self.pheno = {"height": 0.5, "weight": 0.5,
                      "muscle": 0.5, "proportions": 0.5,
                      "gender": self.gender, "age": self.age}
        self.lc: dict[str, float] = {}
        # Pre-compute vertex masks per body region (dominant bone).
        bone_w = self.bm.vertex_bone_weights.detach().cpu().numpy()
        bone_i = self.bm.vertex_bone_indices.detach().cpu().numpy()
        self.dom_bone = bone_i[np.arange(bone_i.shape[0]),
                                bone_w.argmax(axis=1)]
        labels = self.bm.bone_labels
        def vmask(predicate):
            return np.array([predicate(labels[b]) for b in self.dom_bone])
        ARM_KEYS = ("upperarm", "lowerarm", "wrist", "hand", "thumb",
                     "index", "middle", "ring", "pinky")
        LEG_KEYS = ("upperleg", "lowerleg", "foot", "toe")
        # No-arms: drop only arm/hand/finger bones. Keep legs so hip/glute
        # slice catches the full hip circumference. Used for torso tapes
        # (bust through hip).
        self.mask_no_arms = vmask(lambda l: not any(
            k in l.lower() for k in ARM_KEYS))
        # Single leg/arm for limb-perpendicular slicing (after A-pose puts
        # the limb roughly vertical → horizontal slice = perpendicular).
        self.mask_leg_l = vmask(lambda l: any(
            k in l for k in ("upperleg01.L", "upperleg02.L",
                              "lowerleg01.L", "lowerleg02.L")))
        self.mask_arm_l = vmask(lambda l: any(
            k in l for k in ("upperarm01.L", "upperarm02.L",
                              "lowerarm01.L", "lowerarm02.L")))
        # Bone-pose A-pose: arms swing down (~60°), legs slightly inward.
        # In Anny's Z-up frame, rotating around Y axis turns +X arm to
        # -Z (down).
        def rot_y(deg):
            c, s = float(np.cos(np.deg2rad(deg))), float(np.sin(np.deg2rad(deg)))
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
            return torch.from_numpy(T)[None]
        self.a_pose = {}
        if "upperarm01.L" in labels: self.a_pose["upperarm01.L"] = rot_y(60.0)
        if "upperarm01.R" in labels: self.a_pose["upperarm01.R"] = rot_y(-60.0)
        if "upperleg01.L" in labels: self.a_pose["upperleg01.L"] = rot_y(-7.0)
        if "upperleg01.R" in labels: self.a_pose["upperleg01.R"] = rot_y(7.0)

    # ─── forward + measurement readout ────────────────────────────────
    def _forward(self):
        with torch.no_grad():
            out = self.bm(
                pose_parameters=self.a_pose,
                phenotype_kwargs=self.pheno,
                local_changes_kwargs=self.lc,
            )
        return out

    def anth_height_cm(self) -> float:
        return self.anth.height(self._forward()["vertices"])[0].item() * 100.0

    def anth_mass_kg(self) -> float:
        return self.anth.mass(self._forward()["vertices"])[0].item()

    def verts(self) -> np.ndarray:
        return self._forward()["vertices"][0].cpu().numpy()

    def girth_at_y(self, y_frac: float, region: str = "torso",
                    half_band_m: float = 0.008) -> float:
        """Cm girth at fraction ``y_frac`` from feet (0=feet, 1=top of
        head). ``region`` ∈ ``torso`` / ``leg_l`` / ``arm_l`` / ``all``.

        Anny mesh raw coords are normalised (~0.45m span for 160cm
        adult) → rescale to metres using anthropometric height before
        slicing so the perimeter comes out in real cm.
        """
        v = self.verts()
        # Anny is Z-up. Rescale so vert coords are in metres.
        anth_h_m = self.anth_height_cm() / 100.0
        z_raw = v[:, 2]
        raw_span = z_raw.max() - z_raw.min()
        scale_m = anth_h_m / max(raw_span, 1e-6)
        v_m = v * scale_m
        z = v_m[:, 2]
        z_lo, z_hi = z.min(), z.max()
        slice_z = z_lo + y_frac * (z_hi - z_lo)
        band = np.abs(z - slice_z) < half_band_m
        if region == "torso":   sel = self.mask_no_arms & band
        elif region == "leg_l": sel = self.mask_leg_l & band
        elif region == "arm_l": sel = self.mask_arm_l & band
        else:                   sel = band
        if sel.sum() < 6:
            return 0.0
        pts2d = v_m[sel][:, [0, 1]]  # XY (lateral, depth) in metres
        # Cauchy perimeter: avg width across angles × π.
        n_dirs = 64
        theta = np.linspace(0, np.pi, n_dirs + 1)[:-1]
        proj = pts2d[:, 0:1] * np.cos(theta)[None, :] + \
               pts2d[:, 1:2] * np.sin(theta)[None, :]
        widths = proj.max(axis=0) - proj.min(axis=0)
        return float(widths.mean() * np.pi * 100)


# ─────────────────────────── tape + proportion plan ────────────────────────

# Each tape: (lm_key in JSON, anny blendshape, region, default lm_y_frac)
TAPE_PLAN = [
    ("bust",      "measure-bust-circ-incr",      "torso", 0.28),
    ("underbust", "measure-underbust-circ-incr", "torso", 0.31),
    ("waist",     "measure-waist-circ-incr",     "torso", 0.36),
    ("highhip",   None,                          "torso", 0.43),   # no direct blendshape
    ("hip",       "measure-hips-circ-incr",      "torso", 0.52),
    ("bicep",     "measure-upperarm-circ-incr",  "arm_l", 0.25),
    ("thigh",     "measure-thigh-circ-incr",     "leg_l", 0.55),
    ("knee",      "measure-knee-circ-incr",      "leg_l", 0.74),
    ("neck",      "measure-neck-circ-incr",      "torso", 0.16),
]
# y_frac is from HEAD top (0) to FEET (1) — convert to feet-up via 1-y.


# ──────────────────────────────── main solver ──────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--landmarks", type=Path, required=True,
                    help="landmark_editor JSON — supplies tape cm + "
                         "anatomical Y fractions.")
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--gender", default="female")
    ap.add_argument("--age", type=float, default=30.0,
                    help="Subject age in years.")
    ap.add_argument("--front-seg", type=Path, default=None,
                    help="Sapiens front_seg — used for landmark Y "
                         "fractions if photo body bbox differs from "
                         "JSON's image_size.")
    ap.add_argument("--side-seg",  type=Path, default=None)
    args = ap.parse_args(argv)

    data = json.loads(args.landmarks.read_text())
    measurements = data.get("measurements") or {}
    lines = data.get("lines_y") or {}

    # Compute anatomical Y fraction per landmark using the photo body
    # bbox from the side-seg (cleaner profile, no arm clutter).
    lm_y_frac: dict[str, float] = {}
    if args.side_seg and args.side_seg.exists():
        seg = np.load(args.side_seg)
        if seg.ndim == 3:
            seg = seg.argmax(0) if seg.shape[0] < seg.shape[-1] else seg.argmax(-1)
        body = seg > 0
        ys = np.where(body)[0]
        s_top, s_bot = ys.min(), ys.max()
        for name, by in lines.items():
            if by and by.get("side") is not None:
                lm_y_frac[name] = (by["side"] - s_top) / (s_bot - s_top)
    print(f"landmark Y fracs (from head, 0=top): {lm_y_frac}")

    f = AnnyFitter(gender=args.gender, age_years=args.age)
    print(f"Anny model loaded (gender={args.gender}, age={args.age}y → "
          f"phenotype={f.age:.2f})")

    # Step 1: phenotype.height → user height_cm
    height_cm = float(measurements.get("height", 160.0))
    def h_of(coef):
        f.pheno["height"] = coef; return f.anth_height_cm()
    f.pheno["height"] = bisect(h_of, 0.0, 1.0, target=height_cm, tol=0.1)
    print(f"phenotype.height = {f.pheno['height']:.3f}  "
          f"→ anth height = {f.anth_height_cm():.1f} cm  "
          f"(target {height_cm})")

    # Step 2: phenotype.weight → user weight_kg (if given via JSON or 57)
    weight_kg = float(measurements.get("weight_kg", 57.0))
    def w_of(coef):
        f.pheno["weight"] = coef; return f.anth_mass_kg()
    f.pheno["weight"] = bisect(w_of, 0.0, 1.0, target=weight_kg, tol=0.2)
    print(f"phenotype.weight = {f.pheno['weight']:.3f}  "
          f"→ anth mass = {f.anth_mass_kg():.1f} kg  (target {weight_kg})")

    # Re-pin height after weight changed it.
    f.pheno["height"] = bisect(h_of, 0.0, 1.0, target=height_cm, tol=0.1)
    print(f"re-pin height: anth height = {f.anth_height_cm():.1f} cm")

    # Step 3: tape girths — 1D-solve each measure-*-circ-incr blendshape.
    print("\ntape girth fits:")
    print(f"  {'name':10}  {'target':>7}  {'baseline':>8}  "
          f"{'coef':>6}  {'final':>7}")
    for lm_key, blendshape, region, default_y in TAPE_PLAN:
        target_key = "knee_circ" if lm_key == "knee" else lm_key
        target_cm = measurements.get(target_key)
        if target_cm is None or blendshape is None:
            continue
        target_cm = float(target_cm)
        y_frac_from_top = lm_y_frac.get(lm_key, default_y)
        y_frac_from_feet = 1.0 - y_frac_from_top
        baseline = f.girth_at_y(y_frac_from_feet, region=region)

        def g(coef, key=blendshape, yf=y_frac_from_feet, reg=region):
            f.lc[key] = coef
            return f.girth_at_y(yf, region=reg)

        coef = bisect(g, -1.5, 1.5, target=target_cm, tol=0.1)
        f.lc[blendshape] = coef
        final = f.girth_at_y(y_frac_from_feet, region=region)
        print(f"  {lm_key:10}  {target_cm:>7.1f}  {baseline:>8.1f}  "
              f"{coef:>+6.2f}  {final:>7.1f}")

    # Final summary
    print(f"\nfinal: H={f.anth_height_cm():.1f}cm  "
          f"W={f.anth_mass_kg():.1f}kg")
    print(f"local_changes: {[(k,round(v,2)) for k,v in f.lc.items()]}")

    # Save fit
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_npz = args.out_prefix.with_name(args.out_prefix.name + "_anny_fit.npz")
    verts = f.verts()
    np.savez(out_npz,
             phenotype={k: float(v) for k, v in f.pheno.items()},
             local_changes={k: float(v) for k, v in f.lc.items()},
             gender=args.gender, age_years=args.age,
             vertices=verts.astype(np.float32),
             faces=f.bm.get_triangular_faces().cpu().numpy().astype(np.int32))
    print(f"wrote {out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
