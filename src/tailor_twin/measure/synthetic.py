"""Synthetic-body validation harness (ROADMAP Workstream A).

Landmark rules and measurement recipes were historically validated
against one person's tape numbers. SMPL-X can *generate* bodies, so
extractor changes can be regression-tested across the shape space with
no capture and no tape:

  1. Sample N bodies (fixed seed; first ``k`` betas ~ N(0,1), clipped)
     in the canonical A-pose — the same pose the pipeline measures in.
  2. Run the real Seamly extractor on each.
  3. Snapshot ``{body: {code: value}}`` to JSON. Later runs compare
     against the snapshot within a tolerance and list per-code diffs.

This is NOT a truth test (the extractor grades its own homework on
absolute values) — it is (a) an unintended-change detector across many
body shapes, (b) a does-every-landmark-rule-succeed gate (dynamic
searches have historically thrown on unusual figures), and (c) a
smoothness check (``--perturb``): a tiny beta jitter must not make any
code jump, which catches landmark rules that flip between vertices.

Each body also records its mesh volume — a weight proxy that feeds the
Workstream D plausibility prior (see references/anthropometry/README.md,
Option 1) without a separate pass.

Torch/smplx are imported lazily inside the functions that need them, so
the sampling + snapshot-compare logic stays unit-testable without the
ML stack. Run via ``scripts/validate_synthetic.py`` on a machine with
the SMPL-X model file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DEFAULT_SNAPSHOT = Path("tests/data/synthetic_snapshot.json")
DEFAULT_NUM_BODIES = 30
DEFAULT_ACTIVE_BETAS = 10
DEFAULT_TOL_CM = 0.05
# Beta clip: |beta| ≤ 2.5 keeps the sample inside the realistic bulk of
# the CAESAR-trained shape space (extreme tails produce non-human shapes
# that would make every landmark rule fail for uninteresting reasons).
BETA_CLIP = 2.5
PERTURB_EPS = 0.05          # beta jitter for the smoothness check
PERTURB_MAX_JUMP_CM = 1.5   # a code moving more than this on eps jitter
#                             indicates a landmark rule snapping between
#                             vertices, not smooth shape response


def sample_betas(
    seed: int,
    num_bodies: int = DEFAULT_NUM_BODIES,
    active: int = DEFAULT_ACTIVE_BETAS,
    num_betas: int = 300,
) -> np.ndarray:
    """(num_bodies, num_betas) float32 — first ``active`` components
    ~ N(0,1) clipped to ±BETA_CLIP, rest zero. Deterministic per seed."""
    rng = np.random.default_rng(seed)
    out = np.zeros((num_bodies, num_betas), dtype=np.float32)
    out[:, :active] = np.clip(
        rng.normal(0.0, 1.0, size=(num_bodies, active)),
        -BETA_CLIP, BETA_CLIP).astype(np.float32)
    return out


def mesh_volume_m3(verts: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume of a closed mesh (divergence theorem; SMPL-X is
    watertight). Density ≈ 1.01 g/cm³ turns this into the weight proxy
    used by the Workstream D plausibility prior."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return float(abs(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum()) / 6.0)


def build_body(
    bm, betas: np.ndarray, apose_deg: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward one SMPL-X body in the canonical A-pose (the pose the
    pipeline measures in — same call shape as ring_deform_cli /
    clean_fit). Returns (verts, joints) float32, canonical frame."""
    import torch

    from ..fit.refine_to_tape import _build_a_pose

    pose = _build_a_pose(apose_deg).astype(np.float32)
    with torch.no_grad():
        out = bm(
            betas=torch.from_numpy(betas[None, :]),
            body_pose=torch.from_numpy(pose.reshape(1, -1)),
            global_orient=torch.zeros(1, 3),
            transl=torch.zeros(1, 3),
        )
    return (out.vertices[0].cpu().numpy().astype(np.float32),
            out.joints[0].cpu().numpy().astype(np.float32))


def extract_body(
    verts: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    gender: str,
) -> tuple[dict[str, float], dict[str, str]]:
    """Run the real Seamly extractor — same call shape as measure/cli.py."""
    from .landmarks import build_landmark_set
    from .seamly_extractor import extract_catalog

    lm = build_landmark_set(verts, joints=joints, faces=faces, gender=gender)
    cat = extract_catalog(verts, faces, joints=joints, gender=gender,
                          landmarks=lm)
    values = {k: round(float(v), 4) for k, v in cat.values.items()}
    skipped = {k: str(r) for k, r in cat.skipped.items()}
    return values, skipped


def run_harness(
    *,
    model_folder: str = "data/body_models",
    gender: str = "female",
    num_bodies: int = DEFAULT_NUM_BODIES,
    active_betas: int = DEFAULT_ACTIVE_BETAS,
    num_betas: int = 300,
    seed: int = 0,
    apose_deg: float = 30.0,
    perturb: bool = False,
    progress: bool = True,
) -> dict:
    """Extract the catalog for every sampled body; return the report dict
    (the thing that gets snapshotted).

    ``perturb=True`` additionally re-extracts each body with every active
    beta nudged by +PERTURB_EPS and records the max per-code jump —
    the smoothness check. Doubles the runtime.
    """
    import smplx

    bm = smplx.create(
        model_path=model_folder, model_type="smplx", gender=gender,
        num_betas=num_betas, use_pca=False, batch_size=1,
    )
    faces = np.asarray(bm.faces, dtype=np.int32)
    all_betas = sample_betas(seed, num_bodies, active_betas, num_betas)

    bodies: list[dict] = []
    for i, betas in enumerate(all_betas):
        verts, joints = build_body(bm, betas, apose_deg)
        values, skipped = extract_body(verts, faces, joints, gender)
        rec: dict = {
            "betas_active": [round(float(b), 6) for b in betas[:active_betas]],
            "volume_m3": round(mesh_volume_m3(
                verts.astype(np.float64), faces), 6),
            "values": values,
            "skipped": skipped,
        }
        if perturb:
            jit = betas.copy()
            jit[:active_betas] += PERTURB_EPS
            v2, j2 = build_body(bm, jit, apose_deg)
            values2, _ = extract_body(v2, faces, j2, gender)
            jumps = {
                code: round(abs(values2[code] - val), 3)
                for code, val in values.items()
                if code in values2
                and abs(values2[code] - val) > PERTURB_MAX_JUMP_CM
            }
            rec["perturb_jumps_cm"] = jumps
        bodies.append(rec)
        if progress:
            n_skip = len(skipped)
            print(f"  body {i + 1:2d}/{num_bodies}: "
                  f"{len(values)} codes, {n_skip} skipped"
                  + (f", {len(rec.get('perturb_jumps_cm', {}))} jump(s)"
                     if perturb else ""))

    return {
        "meta": {
            "gender": gender,
            "num_bodies": num_bodies,
            "active_betas": active_betas,
            "num_betas": num_betas,
            "seed": seed,
            "apose_deg": apose_deg,
            "beta_clip": BETA_CLIP,
        },
        "bodies": bodies,
    }


def compare_reports(
    baseline: dict,
    current: dict,
    tol_cm: float = DEFAULT_TOL_CM,
) -> list[str]:
    """Diff two harness reports; return human-readable failure lines
    (empty = pass). Pure dict logic — unit-tested without the ML stack.

    Checks, per body: value drift beyond ``tol_cm``, codes that appeared
    or disappeared, and newly skipped codes. A meta mismatch (different
    seed/sample) fails immediately — the snapshots aren't comparable.
    """
    failures: list[str] = []
    if baseline.get("meta") != current.get("meta"):
        return [f"meta mismatch — snapshots not comparable:\n"
                f"  baseline: {baseline.get('meta')}\n"
                f"  current:  {current.get('meta')}"]
    b_bodies = baseline.get("bodies", [])
    c_bodies = current.get("bodies", [])
    if len(b_bodies) != len(c_bodies):
        return [f"body-count mismatch: baseline={len(b_bodies)} "
                f"current={len(c_bodies)}"]
    for i, (b, c) in enumerate(zip(b_bodies, c_bodies)):
        bv, cv = b["values"], c["values"]
        for code in sorted(set(bv) - set(cv)):
            failures.append(
                f"body {i}: {code} disappeared "
                f"(now skipped: {c['skipped'].get(code, 'no reason')!r})")
        for code in sorted(set(cv) - set(bv)):
            failures.append(f"body {i}: {code} newly appeared "
                            f"({cv[code]:.2f} cm)")
        for code in sorted(set(bv) & set(cv)):
            d = cv[code] - bv[code]
            if abs(d) > tol_cm:
                failures.append(
                    f"body {i}: {code} drifted {d:+.3f} cm "
                    f"({bv[code]:.3f} → {cv[code]:.3f})")
    return failures


def load_snapshot(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def save_snapshot(report: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, sort_keys=True))
