"""Port an SMPL-X mesh to Anny via Anny's built-in ParametersRegressor.

Anny ships with a vertex-to-parametric regressor that estimates Anny
phenotype + pose to best match a target mesh. Per the Anny paper,
mean cyclic SMPL-X ↔ Anny error is ~3 mm.

Usage:
  python smplx_to_anny.py \
      --smplx-npz pair_..._lm23_smplx_fit.npz \
      --out-prefix pair_..._lm23_anny
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import anny


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smplx-npz", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--gender", default="female")
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--n-points", type=int, default=5000)
    args = ap.parse_args(argv)

    # Load SMPL-X verts.
    z = np.load(args.smplx_npz, allow_pickle=True)
    smplx_v = z["smplx_vertices"]
    print(f"smplx verts: {smplx_v.shape}  "
          f"Y span={smplx_v[:,1].max()-smplx_v[:,1].min():.3f}m")

    # Build Anny model with SMPL-X topology so output verts share index
    # space with the SMPL-X target — needed for the ParametersRegressor
    # which uses per-bone vertex indices for joint-wise registration.
    torch.set_default_dtype(torch.float32)
    from anny.models.retopology import create_smplx_topology_model
    bm = create_smplx_topology_model(
        local_changes="all", all_phenotypes=False).to(torch.float32)
    print(f"Anny (SMPL-X topology): "
          f"{bm.template_vertices.shape[0]} verts")
    reg = anny.ParametersRegressor(bm, n_points=args.n_points,
                                    max_n_iters=args.max_iters,
                                    verbose=True)

    # Wrap target as (1, N, 3) tensor.
    target = torch.from_numpy(smplx_v).float().unsqueeze(0)

    # Lock gender (we know it).
    init_pheno = {
        "gender": torch.tensor([0.0 if args.gender == "female" else 1.0]),
    }

    print("\nfitting Anny to SMPL-X target …")
    pose, pheno, v_hat = reg.fit_with_age_anchor_search(
        vertices_target=target,
        initial_phenotype_kwargs=init_pheno,
        excluded_phenotypes=["gender"],  # don't optimize gender
    )

    print(f"\nfitted phenotype:")
    for k, v in pheno.items():
        print(f"  {k:15} = {float(v):.3f}")

    verts_out = v_hat[0].detach().cpu().numpy()
    faces = bm.get_triangular_faces().detach().cpu().numpy()

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_npz = args.out_prefix.with_name(args.out_prefix.name + "_anny.npz")
    np.savez(out_npz,
             phenotype={k: float(v) for k, v in pheno.items()},
             vertices=verts_out.astype(np.float32),
             faces=faces.astype(np.int32),
             pose=pose.detach().cpu().numpy())
    print(f"wrote {out_npz}")

    out_obj = args.out_prefix.with_name(args.out_prefix.name + "_anny.obj")
    with open(out_obj, "w") as fh:
        fh.write("# Anny mesh ported from SMPL-X via ParametersRegressor\n")
        for x, y, zz in verts_out:
            fh.write(f"v {x:.6f} {y:.6f} {zz:.6f}\n")
        for a, b, c in faces:
            fh.write(f"f {a+1} {b+1} {c+1}\n")
    print(f"wrote {out_obj}")

    # Reconstruction error.
    # Compare nearest-neighbor distances (no topology correspondence).
    src = torch.from_numpy(smplx_v).float()
    dst = torch.from_numpy(verts_out).float()
    d_sd = torch.cdist(src, dst).min(dim=1).values
    d_ds = torch.cdist(dst, src).min(dim=1).values
    print(f"\nnearest-neighbor distance (mm):")
    print(f"  SMPLX → Anny  mean={d_sd.mean()*1000:.2f}  "
          f"max={d_sd.max()*1000:.2f}")
    print(f"  Anny  → SMPLX mean={d_ds.mean()*1000:.2f}  "
          f"max={d_ds.max()*1000:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
