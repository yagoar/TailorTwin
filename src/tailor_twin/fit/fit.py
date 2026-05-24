"""SMPL-X mesh-to-mesh fitting.

Adapted from SMPLify-X (https://github.com/vchoutas/smplify-x) for the case
where the input is a target mesh (e.g. a LiDAR scan reconstruction) rather
than an image plus 2D keypoints. The 2D keypoint loss is replaced with
bidirectional chamfer; everything else follows SMPLify-X conventions:

  - LBFGS with strong-Wolfe line search
  - staged unfreezing (rigid -> +shape -> +pose-via-VPoser -> relaxed)
  - L2 shape prior, L2 on VPoser latent z (SMPLify-X: pose_embedding.pow(2).sum())
  - elbow/knee angle prior (SMPLify-X angle_prior)

See GUARDRAILS.md sections 1.2 and 4 for the rules governing this code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import smplx
import torch
from scipy.spatial import cKDTree

from . import align
from .losses import (
    angle_prior_elbow_knee,
    chamfer_bidirectional,  # noqa: F401  — kept for backwards compatibility
    chamfer_point_to_surface,
    chamfer_scan_to_mesh,
    crop_scan_for_chamfer,
    displacement_l2,
    displacement_laplacian,
    pose_prior_l2,
    pose_prior_z_l2,
    shape_prior,
)
from .vposer import VPoserWrapper


@dataclass
class FitConfig:
    model_folder: str = "data/body_models"
    gender: str = "female"
    num_betas: int = 300
    device: str = "cpu"
    # SMPL-X+D adds per-vertex displacement (stages 7-9). Off by default —
    # the high-beta parametric fit is smoother and matches the addon mesh
    # fidelity. Re-enable when fit residual matters more than smoothness.
    use_displacement: bool = False
    # If True, run ONLY stage 7 of the D series (heavy Laplacian smoothing,
    # iters bumped). Captures broad shape error (e.g. pelvis bulge) without
    # baking high-frequency scan noise into D. Implies use_displacement.
    use_smooth_displacement: bool = False
    # Partial-cloud mode: a single-view pointmap is only a front (or side)
    # shell. Use a one-directional scan→mesh chamfer so the unobserved
    # mesh regions are not collapsed toward the seen surface.
    partial_cloud: bool = False
    # Lock the body to a canonical A-pose (shoulder angle, degrees) and
    # fit only global_orient + transl + betas — never body_pose. The
    # chamfer cloud must be arm-free; the result is a consistent,
    # pose-normalized A-pose body for measurement.
    canonical_pose_deg: float | None = None
    vposer_ckpt: str | None = "data/vposer/vposer_v1_0/snapshots/TR00_E096.pt"
    # Crop hair/floor from the chamfer target — see losses.crop_scan_for_chamfer.
    crop_above_y_frac: float = 0.97
    crop_below_y_frac: float = 0.02
    # Stage weights. Chamfer always 1; priors scaled relative to it.
    stage_weights: list[dict] = field(default_factory=lambda: [
        # Stage 1 — rigid: global_orient + transl
        dict(chamfer=1.0, shape=0.0, pose=0.0, angle=0.0, iters=20,
             unfreeze=("global_orient", "transl"), use_vposer=False),
        # Stage 2 — +shape: pelvis-anchored shape adjustment
        dict(chamfer=1.0, shape=0.01, pose=0.0, angle=0.0, iters=30,
             unfreeze=("global_orient", "transl", "betas"), use_vposer=False),
        # Stage 3 — +pose via VPoser z (32D latent), strong z prior
        dict(chamfer=1.0, shape=0.005, pose=0.01, angle=10.0, iters=60,
             unfreeze=("global_orient", "transl", "betas", "z"), use_vposer=True),
        # Stage 4 — relax z prior
        dict(chamfer=1.0, shape=0.002, pose=0.001, angle=10.0, iters=60,
             unfreeze=("global_orient", "transl", "betas", "z"), use_vposer=True),
        # Stage 5 — release into raw body_pose with very loose prior.
        dict(chamfer=1.0, shape=0.0001, pose=0.0002, angle=10.0, iters=80,
             unfreeze=("global_orient", "transl", "betas", "body_pose"),
             use_vposer=False),
        # Stage 6 — final tighten. Safe to push priors lower with
        # point-to-surface chamfer (no inside-collapse).
        dict(chamfer=1.0, shape=0.00002, pose=0.00005, angle=10.0, iters=120,
             unfreeze=("global_orient", "transl", "betas", "body_pose"),
             use_vposer=False),
        # Stage 7 — SMPL-X+D: add per-vertex displacement field D.
        # Chamfer weight bumped 100x because by this stage the per-vertex
        # mean squared distance is ~1e-4 m² and we need LBFGS gradients
        # large enough to actually move D away from zero. Heavy Laplacian
        # smoothness keeps D absorbing broad shape error, not scan noise.
        dict(chamfer=100.0, shape=0.0, pose=0.0, angle=0.0, iters=60,
             d_l2=0.1, d_lap=50.0,
             unfreeze=("d",), use_vposer=False),
        # Stage 8 — release more DOF: relax Laplacian, fine-tune D.
        dict(chamfer=100.0, shape=0.0, pose=0.0, angle=0.0, iters=80,
             d_l2=0.01, d_lap=5.0,
             unfreeze=("d",), use_vposer=False),
        # Stage 9 — final D polish: loose Laplacian, longer.
        dict(chamfer=100.0, shape=0.0, pose=0.0, angle=0.0, iters=120,
             d_l2=0.001, d_lap=0.5,
             unfreeze=("d",), use_vposer=False),
    ])


@dataclass
class FitResult:
    betas: np.ndarray
    body_pose: np.ndarray  # axis-angle (21, 3)
    global_orient: np.ndarray  # axis-angle (3,)
    transl: np.ndarray
    z: np.ndarray | None  # VPoser latent if used in final stage
    smplx_vertices: np.ndarray
    smplx_joints: np.ndarray
    final_chamfer: float
    displacement: np.ndarray | None = None  # per-vertex D (10475, 3)
    gender: str = "female"  # one of: female / male / neutral. Persisted
    # in the fit npz so the viewer + bent-arm re-pose pick the matching
    # SMPL-X model file at load time (legacy fits without this key
    # fall back to female).
    # Subject identity (persisted in the npz so the SMIS export can fill
    # the <personal> block without re-passing --person-* args every time
    # the measure CLI runs). Empty strings round-trip as missing fields.
    person_given_name: str = ""
    person_family_name: str = ""
    person_birth_date: str = ""  # ISO yyyy-mm-dd


def _set_requires_grad(body_model, transl, z, d, names: tuple[str, ...]):
    for p in body_model.parameters():
        p.requires_grad_(False)
    if z is not None:
        z.requires_grad_(False)
    if d is not None:
        d.requires_grad_(False)
    for n in names:
        if n == "transl":
            transl.requires_grad_(True)
        elif n == "z":
            z.requires_grad_(True)
        elif n == "d":
            d.requires_grad_(True)
        else:
            getattr(body_model, n).requires_grad_(True)


def fit_scan(
    scan_verts: np.ndarray,
    cfg: FitConfig | None = None,
    verbose: bool = True,
    scan_faces: np.ndarray | None = None,
) -> FitResult:
    cfg = cfg or FitConfig()
    device = torch.device(cfg.device)

    body_model = smplx.create(
        model_path=cfg.model_folder,
        model_type="smplx",
        gender=cfg.gender,
        num_betas=cfg.num_betas,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    ).to(device)

    # Canonical A-pose: lock body_pose, never optimize it. Restrict the
    # chamfer to non-arm vertices — the cloud is arm-free, so an arm
    # vertex must not be allowed to snap onto a torso scan point.
    mesh_keep = None
    if cfg.canonical_pose_deg is not None:
        from .refine_to_tape import _build_a_pose
        ap = _build_a_pose(cfg.canonical_pose_deg).reshape(1, -1)
        body_model.body_pose = torch.nn.Parameter(
            torch.from_numpy(ap.astype(np.float32)).to(device),
            requires_grad=False)
        from ..measure.regions import region_vertex_mask
        arm = (region_vertex_mask(("left_arm",),
                                  model_folder=cfg.model_folder,
                                  gender=cfg.gender)
               | region_vertex_mask(("right_arm",),
                                    model_folder=cfg.model_folder,
                                    gender=cfg.gender))
        mesh_keep = torch.from_numpy(~arm).to(device)

    vposer = None
    if cfg.canonical_pose_deg is not None:
        pass  # pose locked — VPoser not used
    elif cfg.vposer_ckpt is not None and Path(cfg.vposer_ckpt).exists():
        vposer = VPoserWrapper(cfg.vposer_ckpt, device=cfg.device)
        if verbose:
            print(f"loaded VPoser from {cfg.vposer_ckpt}")
    elif verbose:
        print("VPoser checkpoint not found — falling back to L2 pose prior")

    z = (
        torch.zeros(1, vposer.latent_dim, device=device, requires_grad=False)
        if vposer is not None
        else None
    )
    if z is not None:
        z = torch.nn.Parameter(z)

    # Per-vertex displacement field D (SMPL-X+D). Init zeros.
    d_param = torch.nn.Parameter(
        torch.zeros(body_model.v_template.shape[0], 3, device=device)
    )

    # Crop chamfer target: drop hair/floor.
    keep_mask = crop_scan_for_chamfer(
        scan_verts, cfg.crop_above_y_frac, cfg.crop_below_y_frac
    )
    scan_cropped = scan_verts[keep_mask].astype(np.float32)
    if verbose:
        print(
            f"scan crop: kept {keep_mask.sum()}/{len(scan_verts)} verts "
            f"(Y range {scan_cropped[:,1].min():.3f}..{scan_cropped[:,1].max():.3f})"
        )

    # Initial centroid alignment uses the cropped scan (avoids hair pulling
    # the initial transl upward).
    with torch.no_grad():
        canon = body_model().vertices[0].cpu().numpy()
    transl_init = align.initial_transl(scan_cropped, canon).astype(np.float32)
    transl = torch.nn.Parameter(
        torch.from_numpy(transl_init).unsqueeze(0).to(device)
    )
    body_model.transl = transl

    scan_t = torch.from_numpy(scan_cropped).to(device)
    scan_tree = cKDTree(scan_cropped)

    # Cotangent Laplacian over the SMPL-X canonical mesh (needed for the
    # D smoothness regularizer). Computed once.
    import potpourri3d as pp3d
    smplx_template = body_model.v_template.detach().cpu().numpy().astype(np.float64)
    smplx_faces_np = body_model.faces.astype(np.int64)
    L_sparse = pp3d.cotan_laplacian(smplx_template, smplx_faces_np).tocoo()
    lap_idx = torch.from_numpy(np.vstack([L_sparse.row, L_sparse.col])).long().to(device)
    lap_val = torch.from_numpy(L_sparse.data.astype(np.float32)).to(device)
    if verbose:
        print(f"built cotangent Laplacian: {L_sparse.nnz} non-zeros over "
              f"{smplx_template.shape[0]} verts")

    # Build a RaycastingScene over the scan triangle mesh so the chamfer can
    # use point-to-surface distance (instead of point-to-nearest-vertex).
    # We need the original scan triangles for this, not just the cropped
    # vertex subset — load via the caller-provided scan or derive from the
    # cropped data. Caller provides only verts; faces come via scan_faces
    # if supplied; otherwise we fall back to point-to-point chamfer.
    import open3d as o3d
    scan_scene = None
    if scan_faces is not None:
        m = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(scan_verts.astype("float32"),
                            dtype=o3d.core.Dtype.Float32),
            o3d.core.Tensor(scan_faces.astype("int32"),
                            dtype=o3d.core.Dtype.UInt32),
        )
        scan_scene = o3d.t.geometry.RaycastingScene()
        scan_scene.add_triangles(m)
        if verbose:
            print(f"built scan raycasting scene: {len(scan_verts)}v {len(scan_faces)}f")

    final_loss = float("inf")

    stages = cfg.stage_weights
    if cfg.canonical_pose_deg is not None:
        # Pose is locked — fit rigid alignment then shape only.
        stages = [
            dict(chamfer=1.0, shape=0.0, pose=0.0, angle=0.0, iters=25,
                 unfreeze=("global_orient", "transl"), use_vposer=False),
            dict(chamfer=1.0, shape=0.01, pose=0.0, angle=0.0, iters=40,
                 unfreeze=("global_orient", "transl", "betas"),
                 use_vposer=False),
            dict(chamfer=1.0, shape=0.001, pose=0.0, angle=0.0, iters=90,
                 unfreeze=("global_orient", "transl", "betas"),
                 use_vposer=False),
        ]
    elif cfg.use_smooth_displacement:
        # Keep parametric stages + ONLY the first D stage (heavy Lap smooth).
        # Bump iters to converge under heavy smoothness regularizer.
        param_stages = [s for s in stages if "d" not in s.get("unfreeze", ())]
        d_stages = [s for s in stages if "d" in s.get("unfreeze", ())]
        if d_stages:
            # Bump chamfer 10x relative to the original stage 7, keep
            # smoothness moderate. Goal: pull D far enough to absorb the
            # broad parametric residual (pelvis, calf bulges) without
            # accreting high-frequency scan noise.
            heavy = {**d_stages[0], "chamfer": 1000.0, "iters": 200,
                     "d_lap": 10.0, "d_l2": 0.01}
            stages = param_stages + [heavy]
        else:
            stages = param_stages
    elif not cfg.use_displacement:
        # Drop the D stages (any stage that unfreezes "d").
        stages = [s for s in stages if "d" not in s.get("unfreeze", ())]
    for stage_i, sw in enumerate(stages):
        if sw.get("use_vposer", False) and vposer is None:
            # No VPoser available — fall back to optimizing body_pose with L2.
            sw = {**sw, "use_vposer": False,
                  "unfreeze": tuple(
                      "body_pose" if n == "z" else n for n in sw["unfreeze"])}

        _set_requires_grad(body_model, transl, z, d_param, sw["unfreeze"])

        params = [p for p in body_model.parameters() if p.requires_grad]
        if z is not None and z.requires_grad:
            params.append(z)
        if d_param.requires_grad:
            params.append(d_param)

        optimizer = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=20,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-6,
            tolerance_change=1e-9,
        )

        use_vposer = sw.get("use_vposer", False) and vposer is not None

        def closure():
            optimizer.zero_grad()
            if use_vposer:
                body_pose_aa = vposer.decode(z).view(1, 63)
                out = body_model(body_pose=body_pose_aa, return_full_pose=False)
            else:
                out = body_model(return_full_pose=False)
            v = out.vertices[0] + d_param  # SMPL-X + per-vertex displacement
            if cfg.partial_cloud:
                l_chamfer = chamfer_scan_to_mesh(
                    v, scan_t, mesh_keep, scan_tree=scan_tree)
            elif scan_scene is not None:
                l_chamfer = chamfer_point_to_surface(v, scan_scene, scan_t)
            else:
                l_chamfer = chamfer_bidirectional(v, scan_t, scan_tree)
            l_shape = shape_prior(body_model.betas)
            if use_vposer:
                l_pose = pose_prior_z_l2(z)
            else:
                l_pose = pose_prior_l2(body_model.body_pose)
            l_angle = angle_prior_elbow_knee(
                vposer.decode(z).view(1, 63) if use_vposer
                else body_model.body_pose
            )
            l_d_l2 = displacement_l2(d_param) if sw.get("d_l2", 0) else \
                torch.tensor(0.0, device=device)
            l_d_lap = displacement_laplacian(d_param, lap_idx, lap_val) \
                if sw.get("d_lap", 0) else torch.tensor(0.0, device=device)
            loss = (
                sw["chamfer"] * l_chamfer
                + sw["shape"] * l_shape
                + sw["pose"] * l_pose
                + sw["angle"] * l_angle
                + sw.get("d_l2", 0) * l_d_l2
                + sw.get("d_lap", 0) * l_d_lap
            )
            loss.backward()
            closure.last = float(loss.item())
            closure.parts = (
                float(l_chamfer.item()),
                float(l_shape.item()),
                float(l_pose.item()),
                float(l_angle.item()),
                float(l_d_l2.item()),
                float(l_d_lap.item()),
            )
            return loss

        for it in range(sw["iters"]):
            optimizer.step(closure)
            if verbose and (it % 10 == 0 or it == sw["iters"] - 1):
                lc, ls, lp, la, ldl, ldla = closure.parts
                print(
                    f"  stage {stage_i+1} iter {it:3d}  "
                    f"loss={closure.last:.6f}  "
                    f"chamfer={lc:.6f}  shape={ls:.3f}  "
                    f"pose={lp:.3f}  angle={la:.4f}  "
                    f"d_l2={ldl:.4f}  d_lap={ldla:.4f}"
                )
        final_loss = closure.parts[0]
        if verbose:
            print(f"stage {stage_i+1} done — chamfer={final_loss:.6f}")

        # If we just left a VPoser stage and the next stage uses raw
        # body_pose, copy the decoded pose into body_model.body_pose so we
        # start from the converged latent rather than zero.
        next_uses_raw = (
            stage_i + 1 < len(stages)
            and not stages[stage_i + 1].get("use_vposer", False)
            and use_vposer
        )
        if next_uses_raw:
            with torch.no_grad():
                aa = vposer.decode(z).view(1, 63)
                body_model.body_pose[:] = aa

    with torch.no_grad():
        if vposer is not None and stages[-1].get("use_vposer", False):
            body_pose_aa = vposer.decode(z).view(1, 63)
            out = body_model(body_pose=body_pose_aa, return_full_pose=False)
            bp_out = body_pose_aa.detach().cpu().numpy()[0].reshape(21, 3)
        else:
            out = body_model()
            bp_out = body_model.body_pose.detach().cpu().numpy()[0].reshape(21, 3)
        final_verts = (out.vertices[0] + d_param).detach().cpu().numpy()

    return FitResult(
        betas=body_model.betas.detach().cpu().numpy()[0],
        body_pose=bp_out,
        global_orient=body_model.global_orient.detach().cpu().numpy()[0],
        transl=transl.detach().cpu().numpy()[0],
        z=z.detach().cpu().numpy()[0] if z is not None else None,
        smplx_vertices=final_verts,
        smplx_joints=out.joints[0].detach().cpu().numpy(),
        final_chamfer=final_loss,
        displacement=d_param.detach().cpu().numpy(),
        gender=cfg.gender,
    )


def save_fit(result: FitResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        betas=result.betas,
        body_pose=result.body_pose,
        global_orient=result.global_orient,
        transl=result.transl,
        z=result.z if result.z is not None else np.array([]),
        smplx_vertices=result.smplx_vertices,
        smplx_joints=result.smplx_joints,
        final_chamfer=np.array([result.final_chamfer]),
        displacement=(result.displacement if result.displacement is not None
                      else np.zeros((result.smplx_vertices.shape[0], 3),
                                    dtype=np.float32)),
        gender=np.array(result.gender),
        person_given_name=np.array(result.person_given_name),
        person_family_name=np.array(result.person_family_name),
        person_birth_date=np.array(result.person_birth_date),
    )


def fit_person_info(fit: dict | "np.lib.npyio.NpzFile") -> dict[str, str]:
    """Read persisted subject identity (given_name / family_name /
    birth_date) from a fit npz mapping. Returns empty strings for any
    field missing in legacy npz files saved before the field landed."""
    out: dict[str, str] = {}
    for key in ("person_given_name", "person_family_name",
                "person_birth_date"):
        if key not in getattr(fit, "files", fit):
            out[key] = ""
            continue
        v = fit[key]
        out[key] = str(v.item() if hasattr(v, "item") else v)
    return out


def fit_gender(fit: dict | "np.lib.npyio.NpzFile") -> str:
    """Read the persisted gender from a fit npz mapping. Falls back to
    ``"female"`` for legacy fits saved before the gender field landed."""
    if "gender" not in getattr(fit, "files", fit):
        return "female"
    g = fit["gender"]
    return str(g.item() if hasattr(g, "item") else g)
