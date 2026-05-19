"""CLI: import SHAPY regressor output(s) into a tailor-twin fit_npz.

Run SHAPY's ``regressor/demo.py`` on one or more images of the same
subject to produce per-image npzs. Point this command at that folder
(or pass the files directly) and it fuses the betas across views,
writes a canonical-pose tailor-twin fit_npz, plus an obj for visual
inspection, and re-runs the measure CLI to produce the CSV / SMIS /
catalog JSON artefacts the rest of the pipeline consumes.

Example::

    tailor-twin shapy-import \\
        path/to/shapy/output/img_*.npz \\
        --out-prefix data/results/me_shapy \\
        --gender female --num-betas 300

After this, refine against tape:

    tailor-twin refine-tape data/results/me_shapy_smplx_fit.npz \\
        --out-prefix data/results/me_shapy_refined \\
        --target A01=160 --target G03=88 --target G07=70 --target G09=99
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _resolve_inputs(paths: list[Path]) -> list[Path]:
    """Expand any directory entries to all *.npz files inside; keep
    file entries as-is. Sorted for determinism."""
    resolved: list[Path] = []
    for p in paths:
        if p.is_dir():
            resolved.extend(sorted(p.glob("*.npz")))
        elif p.is_file():
            resolved.append(p)
        else:
            raise FileNotFoundError(f"input not found: {p}")
    if not resolved:
        raise ValueError("no SHAPY npz files resolved from inputs")
    return resolved


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("inputs", type=Path, nargs="+",
                   help="SHAPY output npz files OR directories containing them.")
    p.add_argument("--out-prefix", type=Path, required=True,
                   help="Prefix for the produced artefacts, e.g. "
                        "data/results/me_shapy. Files will land at "
                        "<prefix>_smplx_fit.npz, <prefix>_fit_body.obj, "
                        "<prefix>_measurements.csv, <prefix>_seamly_catalog.json.")
    p.add_argument("--gender", default="female",
                   choices=("female", "male", "neutral"))
    p.add_argument("--num-betas", type=int, default=300,
                   help="SMPL-X model beta count used downstream. SHAPY "
                        "outputs 10 betas; the remaining slots are zero-"
                        "padded so the fit matches the model basis the "
                        "measure CLI and refine-tape build.")
    p.add_argument("--model-folder", default="data/body_models")
    p.add_argument("--keep-pose", action="store_true",
                   help="Keep the pose / global_orient / transl from the "
                        "first SHAPY result. Default is canonical T-pose "
                        "(the standard tailor-twin convention).")
    p.add_argument(
        "--no-extract", action="store_true",
        help="Skip the measure CLI re-run. Only writes the fit npz + obj.")
    args = p.parse_args(argv)

    from .shapy_loader import (
        load_shapy_npz,
        shapy_to_fit,
        save_fit_payload,
    )

    inputs = _resolve_inputs(args.inputs)
    print(f"loading {len(inputs)} SHAPY npz(s):")
    results = []
    for inp in inputs:
        r = load_shapy_npz(inp)
        print(f"  {inp.name}  betas[:3]={r.betas[:3]}")
        results.append(r)

    payload = shapy_to_fit(
        results,
        out_num_betas=args.num_betas,
        canonical_pose=not args.keep_pose,
        gender=args.gender,
        model_folder=args.model_folder,
    )

    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fit_npz = out_prefix.with_name(out_prefix.name + "_smplx_fit.npz")
    save_fit_payload(payload, fit_npz)
    print(f"\nwrote {fit_npz}  (multi-view-fused, {args.num_betas} betas, "
          f"gender={args.gender})")

    if args.no_extract:
        return 0

    out_obj = out_prefix.with_name(out_prefix.name + "_fit_body.obj")
    out_csv = out_prefix.with_name(out_prefix.name + "_measurements.csv")
    out_json = out_prefix.with_name(out_prefix.name + "_seamly_catalog.json")
    out_smis = out_prefix.with_name(out_prefix.name + ".smis")
    cmd = [
        sys.executable, "-m", "tailor_twin.measure.cli", str(fit_npz),
        "--num-betas", str(args.num_betas),
        "--gender", args.gender,
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
