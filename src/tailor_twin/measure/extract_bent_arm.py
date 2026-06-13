"""Compute bent-arm measurements (L01, L02, L04) by re-posing the fitted
SMPL-X body with the left elbow flexed.

Thin CLI wrapper around `tailor_twin.measure.bent_arm.repose_bent_arm`,
which is shared with the catalog extractor. Outputs:

  data/results/<basename>_bent_arm.json
    L01 cm  (acromion → wrist, bent)
    L02 cm  (acromion → elbow, bent)
    L03 cm  (= L01 - L02)
    L04 cm  (elbow girth, bent)
  data/results/<basename>_bent_arm.npz
    The bent fit, same key layout as the input npz so the review viewer
    and render_measurement_review.py can load it directly. `displacement`
    is zeroed because it is already baked into `smplx_vertices` here.

Caveats:
  * SMPL-X pose blend shapes handle skin deformation at the elbow but
    do NOT simulate flexed-muscle bulge — elbow girth may be off by
    ~1-2 cm vs a real bent-arm scan.
  * Per-vertex displacement (SMPL-X+D) is reused unchanged. It was fit
    in A-pose, so artefacts on the arm are possible but small.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import smplx

from tailor_twin.measure.bent_arm import (
    DEFAULT_ELBOW_AXIS,
    DEFAULT_ELBOW_FLEX_DEG,
    DEFAULT_SHOULDER_FORWARD_DEG,
    L_ELBOW_BODY_POSE_INDEX,
    L_SHOULDER_BODY_POSE_INDEX,
    repose_bent_arm,
)
from tailor_twin.measure.landmarks import build_landmark_set
from tailor_twin.measure.seamly_catalog import RECIPES


BENT_ARM_CODES: tuple[str, ...] = ("L01", "L02", "L04")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("fit_npz", type=Path)
    p.add_argument("--model-folder", default="data/body_models")
    p.add_argument(
        "--gender", default=None,
        help="SMPL-X model gender. Defaults to the gender persisted in "
             "the fit npz; legacy fits fall back to female.")
    p.add_argument("--num-betas", type=int, default=300)
    p.add_argument(
        "--elbow-flex-deg", type=float, default=DEFAULT_ELBOW_FLEX_DEG,
        help="Elbow flex angle in degrees (rotation magnitude).",
    )
    p.add_argument(
        "--elbow-axis", type=str, default=DEFAULT_ELBOW_AXIS,
        help="Comma-separated 3D axis-angle direction for the elbow "
             "rotation, in the L_Elbow local frame. Default "
             f"{DEFAULT_ELBOW_AXIS!r} bends the forearm forward in front "
             "of the body (Seamly tailoring pose).",
    )
    p.add_argument(
        "--shoulder-forward-deg", type=float,
        default=DEFAULT_SHOULDER_FORWARD_DEG,
        help="Rotate the shoulder forward by this many degrees (around "
             "world X). 0 keeps the fitted A-pose shoulder.",
    )
    p.add_argument(
        "--out-json", type=Path, default=None,
        help="Default: <fit_npz>_bent_arm.json next to the input.",
    )
    args = p.parse_args(argv)

    fit = np.load(args.fit_npz, allow_pickle=True)
    from ..fit.fit import fit_gender
    gender = args.gender or fit_gender(fit)
    body_model = smplx.create(
        model_path=args.model_folder,
        model_type="smplx",
        gender=gender,
        num_betas=args.num_betas,
        use_pca=False,
        batch_size=1,
    )
    faces = np.asarray(body_model.faces, dtype=np.int32)

    pose = repose_bent_arm(
        fit, body_model,
        elbow_flex_deg=args.elbow_flex_deg,
        elbow_axis=args.elbow_axis,
        shoulder_forward_deg=args.shoulder_forward_deg,
    )
    landmarks = build_landmark_set(
        pose.verts, joints=pose.joints, faces=faces, gender=gender,
    )

    # Use the catalog's bent-arm recipes directly so the script's output
    # matches the viewer / render. Catalog L01/L02 route via
    # bicep_max_left so the path stays on the OUTSIDE of the arm.
    values: dict[str, float] = {}
    for code in BENT_ARM_CODES:
        try:
            values[code] = float(
                RECIPES[code].compute(pose.verts, faces, landmarks))
        except Exception as e:  # noqa: BLE001
            values[code] = float("nan")
            print(f"  {code}: ERROR {e}")
    values["L03"] = values["L01"] - values["L02"]

    out_path = (
        args.out_json
        if args.out_json is not None
        else args.fit_npz.with_name(args.fit_npz.stem + "_bent_arm.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fit": str(args.fit_npz),
        "elbow_flex_deg": args.elbow_flex_deg,
        "elbow_axis_angle": pose.elbow_aa.tolist(),
        "values_cm": {k: round(v, 2) for k, v in values.items()},
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")
    for code, val in sorted(values.items()):
        print(f"  {code}: {val:.2f} cm")

    # Save bent mesh as an npz with the same key layout as the input
    # fit so the measure viewer can load it directly. `displacement` is
    # zeroed because it has already been baked into the saved vertices here.
    npz_path = args.fit_npz.with_name(args.fit_npz.stem + "_bent_arm.npz")
    bent_pose = fit["body_pose"].copy()
    bent_pose[L_ELBOW_BODY_POSE_INDEX] = pose.elbow_aa
    bent_pose[L_SHOULDER_BODY_POSE_INDEX] = pose.shoulder_aa
    bent_npz = {
        "betas": fit["betas"],
        "body_pose": bent_pose,
        "global_orient": fit["global_orient"],
        "transl": fit["transl"],
        "smplx_vertices": pose.verts.astype(np.float32),
        "smplx_joints": pose.joints.astype(np.float32),
        "displacement": np.zeros_like(pose.verts, dtype=np.float32),
    }
    if "z" in fit.files:
        bent_npz["z"] = fit["z"]
    if "final_chamfer" in fit.files:
        bent_npz["final_chamfer"] = fit["final_chamfer"]
    np.savez(npz_path, **bent_npz)
    print(f"wrote {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
