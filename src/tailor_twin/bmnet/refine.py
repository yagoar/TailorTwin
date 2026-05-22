"""BMnet girths → SMPL-X betas: pin a base fit to the regressed tape.

The pointmap/silhouette pipeline gives a metrically faithful *front
surface* and a plausible prior-filled back, but circumference under-reads
because a front+side bounding box does not determine the cross-section
shape. BMnet regresses chest/waist/hip girth directly from the same two
silhouettes; this stage feeds those three girths into
``fit.refine_to_tape``, which runs damped Gauss-Newton on the SMPL-X
betas of the base fit until the project's own extractor reports them.

Only the cross-section-ambiguous girths are transferred. Height is a
known input, and BMnet's limb measurements (arm/leg length, bicep…) are
left to the geometry the front pointmap already pinned.

  python -m tailor_twin.bmnet.refine BASE_FIT.npz FRONT_seg.npy SIDE_seg.npy \
      --height 160 --weight 57 --out-prefix data/results/pair_bmnet
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

# BMnet measurement name → seamly extractor code. Matches eval.bodym_eval:
# G04 bust/chest circ, G07 waist circ, G09 hip circ. These three are the
# cross-section-ambiguous girths the silhouette box cannot pin.
GIRTH_MAP = {"chest": "G04", "waist": "G07", "hip": "G09"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("base_fit", type=Path, help="SMPL-X fit npz to refine")
    p.add_argument("front_seg", type=Path, help="front Sapiens '*_seg.npy'")
    p.add_argument("side_seg", type=Path, help="side Sapiens '*_seg.npy'")
    p.add_argument("--height", type=float, required=True)
    p.add_argument("--weight", type=float, required=True)
    p.add_argument("--out-prefix", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, default=Path("data/results/bmnet.pt"))
    p.add_argument("--model-folder", default="data/body_models")
    p.add_argument("--num-betas-active", type=int, default=10)
    p.add_argument("--a-pose-shoulder-deg", type=float, default=30.0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    from .predict import predict_measurements
    from ..fit.silhouette import load_silhouette
    from ..fit.refine_to_tape import refine_betas_to_tape, save_refined_fit

    # Full-body silhouettes (arms kept) — BodyM trains on whole-body
    # A-pose masks, so BMnet expects the arms inside both views.
    front_mask, _ = load_silhouette(str(args.front_seg), arm_classes=())
    side_mask, _ = load_silhouette(str(args.side_seg), arm_classes=())

    pred = predict_measurements(front_mask, side_mask, args.height,
                                args.weight, ckpt=args.ckpt,
                                device=args.device)
    print(f"BMnet measurements (cm)  [ckpt {args.ckpt}]")
    for name, v in pred.items():
        flag = "  ->  " + GIRTH_MAP[name] if name in GIRTH_MAP else ""
        print(f"  {name:<20} {v:7.1f}{flag}")

    targets = {GIRTH_MAP[k]: pred[k] for k in GIRTH_MAP}
    print(f"\nrefining betas to girth targets: "
          f"{ {c: round(v, 1) for c, v in targets.items()} }")

    res = refine_betas_to_tape(
        args.base_fit, targets,
        model_folder=args.model_folder,
        num_betas_active=args.num_betas_active,
        a_pose_shoulder_deg=args.a_pose_shoulder_deg,
    )
    print("\ngirth before -> after (cm):")
    for c in targets:
        print(f"  {c}  {res.values_before[c]:6.1f} -> "
              f"{res.values_after[c]:6.1f}  (target {targets[c]:.1f})")

    out_npz = args.out_prefix.with_name(args.out_prefix.name
                                        + "_smplx_fit.npz")
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    save_refined_fit(args.base_fit, res, out_npz,
                     canonical_pose=True,
                     a_pose_shoulder_deg=args.a_pose_shoulder_deg,
                     model_folder=args.model_folder)
    print(f"wrote {out_npz}")

    fit = np.load(args.base_fit)
    from ..fit.fit import fit_gender
    gender = fit_gender(fit)
    num_betas = int(res.betas.shape[0])
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
