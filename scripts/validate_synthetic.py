#!/usr/bin/env python
"""Synthetic-body validation harness CLI (ROADMAP Workstream A).

Regenerate the committed snapshot (do this ONCE on a known-good tree,
review the diff, commit)::

    .venv/bin/python scripts/validate_synthetic.py --write

Gate a change (compares against the snapshot, exit 1 on drift)::

    .venv/bin/python scripts/validate_synthetic.py

Add the smoothness check (~(active_betas+1)x runtime; flags landmark
rules that snap between vertices under a per-beta jitter)::

    .venv/bin/python scripts/validate_synthetic.py --perturb

Needs the SMPL-X model file (data/body_models/smplx/SMPLX_FEMALE.npz)
and the ML environment — i.e. the user's machine, not the sandbox.
The snapshot must always come from an actual run: never hand-edit it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running straight from a checkout without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tailor_twin.measure.synthetic import (  # noqa: E402
    DEFAULT_ACTIVE_BETAS,
    DEFAULT_NUM_BODIES,
    DEFAULT_SNAPSHOT,
    DEFAULT_TOL_CM,
    compare_reports,
    load_snapshot,
    run_harness,
    save_snapshot,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-folder", default="data/body_models")
    p.add_argument("--gender", default="female")
    p.add_argument("--num-bodies", type=int, default=DEFAULT_NUM_BODIES)
    p.add_argument("--active-betas", type=int, default=DEFAULT_ACTIVE_BETAS)
    p.add_argument("--num-betas", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--apose-deg", type=float, default=30.0)
    p.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    p.add_argument("--tol-cm", type=float, default=DEFAULT_TOL_CM)
    p.add_argument("--write", action="store_true",
                   help="Regenerate the snapshot instead of comparing.")
    p.add_argument("--perturb", action="store_true",
                   help="Also run the per-beta jitter smoothness check "
                        "(~(active_betas+1)x runtime). Worst per-code jump "
                        "and its beta are recorded per body; any jump fails "
                        "the run.")
    args = p.parse_args(argv)

    report = run_harness(
        model_folder=args.model_folder,
        gender=args.gender,
        num_bodies=args.num_bodies,
        active_betas=args.active_betas,
        num_betas=args.num_betas,
        seed=args.seed,
        apose_deg=args.apose_deg,
        perturb=args.perturb,
    )

    rc = 0
    if args.perturb:
        jumpy = [(i, rec["perturb_jumps_cm"])
                 for i, rec in enumerate(report["bodies"])
                 if rec.get("perturb_jumps_cm")]
        if jumpy:
            print(f"\nSMOOTHNESS: {len(jumpy)} body(ies) with code jumps "
                  "under a per-beta +0.05 jitter (landmark rule snapping?):")
            for i, jumps in jumpy:
                pretty = ", ".join(
                    f"{c}={d['cm']}cm@β{d['beta']}"
                    for c, d in sorted(jumps.items(),
                                       key=lambda kv: -kv[1]["cm"]))
                print(f"  body {i}: {pretty}")
            rc = 1
        else:
            print("\nsmoothness: no code jumped > threshold under jitter")

    if args.write:
        save_snapshot(report, args.snapshot)
        print(f"\nwrote {args.snapshot} — review + commit it")
        return rc

    if not args.snapshot.is_file():
        print(f"\nERROR: snapshot missing: {args.snapshot}\n"
              "Generate it once with --write on a known-good tree.")
        return 1
    failures = compare_reports(load_snapshot(args.snapshot), report,
                               tol_cm=args.tol_cm)
    if failures:
        print(f"\n{len(failures)} drift(s) vs {args.snapshot}:")
        for line in failures[:60]:
            print(f"  {line}")
        if len(failures) > 60:
            print(f"  … and {len(failures) - 60} more")
        return 1
    print(f"\nOK: matches {args.snapshot} within ±{args.tol_cm} cm "
          f"({args.num_bodies} bodies)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
