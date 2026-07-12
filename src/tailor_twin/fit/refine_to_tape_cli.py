"""CLI: refine a fit npz's betas against user-supplied tape measurements.

Reads a fit (``--fit fit.npz``) and a list of target codes
(``--target A01=160 --target G07=70 …``), runs damped Gauss-Newton on
the first ``--num-betas-active`` betas to minimise the residual against
the seamly catalog extractor, and writes ``<fit>_refined.npz`` plus a
matching ``_refined.obj`` and ``_refined_measurements.csv``.

Targets are seamly catalog codes (see ``measure/seamly_catalog.py``).
Common ones:

  A01  height
  G02  neck_circ
  G03  highbust_circ
  G07  waist_circ
  G08  highhip_circ
  G09  hip_circ
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_target(spec: str) -> tuple[str, float]:
    if "=" not in spec:
        raise SystemExit(
            f"--target expects 'CODE=value_cm', got {spec!r}")
    code, val = spec.split("=", 1)
    return code.strip(), float(val)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("fit_npz", type=Path)
    p.add_argument("--out-prefix", type=Path, default=None,
                   help="Output prefix. Defaults to <fit_npz with '_refined' suffix>.")
    p.add_argument("--target", action="append", required=True,
                   help="CODE=value_cm, e.g. --target G07=70. Repeatable.")
    p.add_argument("--model-folder", default="data/body_models")
    p.add_argument("--num-betas-active", type=int, default=10,
                   help="How many leading betas to optimise. "
                        "Remaining stay at chamfer-fit values.")
    p.add_argument("--max-iters", type=int, default=12)
    p.add_argument("--tol-cm", type=float, default=0.5)
    p.add_argument("--step-clip", type=float, default=0.4)
    p.add_argument("--anchor-weight", type=float, default=0.05)
    p.add_argument("--ridge", type=float, default=0.01)
    p.add_argument("--waist-y", type=float, default=None,
                   help="World-frame Y override for the waist landmark "
                        "(same semantics as measure/cli.py --waist-y).")
    p.add_argument("--waist-height-cm", type=float, default=None,
                   help="Tape-measured waist height (floor → natural waist, "
                        "vertical, cm). Frame-robust alternative to "
                        "--waist-y: resolved as mesh min Y + height on the "
                        "canonical body at every solver evaluation. "
                        "--waist-y wins if both set.")
    p.add_argument("--a-pose-shoulder-deg", type=float, default=30.0,
                   help="Shoulder rotation (degrees) for the saved "
                        "canonical pose. 0 = T-pose, 30 = standard "
                        "A-pose. Negative flips arm direction if the "
                        "result swings the wrong way.")
    args = p.parse_args(argv)

    targets = dict(_parse_target(s) for s in args.target)

    waist_y_override: float | None = args.waist_y
    if waist_y_override is not None:
        print(f"waist Y override: {waist_y_override:.4f} m")

    from .refine_to_tape import refine_betas_to_tape, save_refined_fit
    res = refine_betas_to_tape(
        args.fit_npz, targets,
        model_folder=args.model_folder,
        num_betas_active=args.num_betas_active,
        max_iters=args.max_iters,
        tol_cm=args.tol_cm,
        step_clip=args.step_clip,
        anchor_weight=args.anchor_weight,
        ridge=args.ridge,
        waist_y_override=waist_y_override,
        waist_height_cm=args.waist_height_cm,
        a_pose_shoulder_deg=args.a_pose_shoulder_deg,
        verbose=True,
    )

    # Resolve output prefix.
    if args.out_prefix is None:
        prefix = args.fit_npz.with_name(
            args.fit_npz.stem.replace("_smplx_fit", "") + "_refined")
    else:
        prefix = args.out_prefix
    out_npz = prefix.with_name(prefix.name + "_smplx_fit.npz")
    out_obj = prefix.with_name(prefix.name + "_fit_body.obj")
    out_csv = prefix.with_name(prefix.name + "_measurements.csv")
    out_json = prefix.with_name(prefix.name + "_seamly_catalog.json")

    save_refined_fit(args.fit_npz, res, out_npz,
                      canonical_pose=True,
                      a_pose_shoulder_deg=args.a_pose_shoulder_deg,
                      model_folder=args.model_folder)
    label = ("T-pose" if abs(args.a_pose_shoulder_deg) < 1e-6
             else f"A-pose ({args.a_pose_shoulder_deg:+.0f}°)")
    print(f"\nwrote {out_npz} (canonical {label})")

    # Re-run the measure CLI on the refined npz to produce the full
    # set of artefacts (csv, json, obj). We forward waist-y if set.
    import subprocess
    cmd = [
        sys.executable, "-m", "tailor_twin.measure.cli", str(out_npz),
        "--save-csv", str(out_csv),
        "--save-seamly-json", str(out_json),
        "--save-obj", str(out_obj),
    ]
    if waist_y_override is not None:
        cmd.extend(["--waist-y", str(waist_y_override)])
    elif args.waist_height_cm is not None:
        cmd.extend(["--waist-height-cm", str(args.waist_height_cm)])
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"measure.cli failed (exit {r.returncode})")
        return r.returncode

    # Print diff summary.
    print("\n=== refinement summary ===")
    print(f"{'code':<6} {'target':>8} {'before':>8} {'after':>8} {'Δ_before':>10} {'Δ_after':>10}")
    for code in res.targets:
        t = res.targets[code]
        b = res.values_before[code]
        a = res.values_after[code]
        print(f"{code:<6} {t:>8.2f} {b:>8.2f} {a:>8.2f} "
              f"{b-t:>+10.2f} {a-t:>+10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
