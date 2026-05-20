"""CLI: fuse a LiDAR-chamfer fit with a SHAPY image-regression fit.

Run after both halves are in ``data/results/``::

    tailor-twin fuse-priors \\
        --lidar data/results/scan_4b750ebd56_smplx_fit.npz \\
        --shapy data/results/me_shapy_smplx_fit.npz \\
        --out-prefix data/results/me_fused \\
        --shapy-weight 0.5

Then refine against tape, same as either parent:

    tailor-twin refine-tape data/results/me_fused_smplx_fit.npz \\
        --out-prefix data/results/me_fused_refined \\
        --target A01=160 --target G03=84 --target G04=88 \\
        --target G07=70 --target G09=99 ...
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lidar", type=Path, required=True,
                   help="LiDAR chamfer fit npz (e.g. scan_XXX_smplx_fit.npz)")
    p.add_argument("--shapy", type=Path, required=True,
                   help="SHAPY-import fit npz (e.g. me_shapy_smplx_fit.npz)")
    p.add_argument("--out-prefix", type=Path, required=True,
                   help="Prefix for fused artefacts.")
    p.add_argument("--shapy-weight", type=float, default=0.5,
                   help="Weight on SHAPY betas in the shared (0-N) block. "
                        "0=LiDAR-only on shared, 1=SHAPY-only on shared.")
    p.add_argument("--n-shared", type=int, default=10,
                   help="Number of leading betas to fuse (default 10 = SHAPY's "
                        "full beta vector). LiDAR keeps the remaining betas[N:].")
    p.add_argument("--a-pose-shoulder-deg", type=float, default=30.0)
    p.add_argument("--model-folder", default="data/body_models")
    p.add_argument("--keep-lidar-displacement", action="store_true",
                   help="Add LiDAR fit's SMPL-X+D displacement to the fused "
                        "mesh. Off by default — clashes with tape refinement.")
    p.add_argument("--no-extract", action="store_true",
                   help="Skip the measure CLI rerun (only write fit npz + obj).")
    args = p.parse_args(argv)

    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fit_npz = out_prefix.with_name(out_prefix.name + "_smplx_fit.npz")
    out_obj = out_prefix.with_name(out_prefix.name + "_fit_body.obj")
    out_csv = out_prefix.with_name(out_prefix.name + "_measurements.csv")
    out_json = out_prefix.with_name(out_prefix.name + "_seamly_catalog.json")
    out_smis = out_prefix.with_name(out_prefix.name + ".smis")

    from .fuse_priors import fuse_to_fit_npz
    fuse_to_fit_npz(
        lidar_npz=args.lidar,
        shapy_npz=args.shapy,
        out_npz=fit_npz,
        shapy_weight=args.shapy_weight,
        n_shared=args.n_shared,
        a_pose_shoulder_deg=args.a_pose_shoulder_deg,
        model_folder=args.model_folder,
        keep_lidar_displacement=args.keep_lidar_displacement,
        verbose=True,
    )

    if args.no_extract:
        return 0

    # Determine num_betas + gender from the fused npz so the measure CLI
    # builds the matching SMPL-X model.
    import numpy as np
    d = np.load(fit_npz)
    n_betas = int(d["betas"].shape[0])
    from .fit import fit_gender
    gender = fit_gender(d)

    cmd = [
        sys.executable, "-m", "tailor_twin.measure.cli", str(fit_npz),
        "--num-betas", str(n_betas),
        "--gender", gender,
        "--model-folder", args.model_folder,
        "--save-csv", str(out_csv),
        "--save-seamly-json", str(out_json),
        "--save-obj", str(out_obj),
        "--save-smis", str(out_smis),
    ]
    r = subprocess.run(cmd)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
