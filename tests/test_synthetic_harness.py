"""Tests for the synthetic-body harness (ROADMAP Workstream A).

The pure parts (beta sampling, volume, snapshot compare) run anywhere.
The full regression gate needs the SMPL-X model file + ML stack + a
committed snapshot — it auto-skips until those exist (generate the
snapshot once with ``scripts/validate_synthetic.py --write``).
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from tailor_twin.measure.synthetic import (
    DEFAULT_SNAPSHOT,
    compare_reports,
    mesh_volume_m3,
    sample_betas,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = REPO_ROOT / "data" / "body_models" / "smplx" / "SMPLX_FEMALE.npz"
SNAPSHOT = REPO_ROOT / DEFAULT_SNAPSHOT


def test_sample_betas_deterministic_and_clipped() -> None:
    a = sample_betas(seed=0, num_bodies=5, active=10, num_betas=300)
    b = sample_betas(seed=0, num_bodies=5, active=10, num_betas=300)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (5, 300)
    assert np.abs(a).max() <= 2.5
    assert not a[:, 10:].any()  # inactive betas stay zero
    c = sample_betas(seed=1, num_bodies=5, active=10, num_betas=300)
    assert not np.array_equal(a, c)


def test_mesh_volume_unit_cube() -> None:
    # Unit cube = 8 verts, 12 triangles, volume 1.
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
                 dtype=np.float64)
    f = np.array([
        [0, 1, 3], [0, 3, 2],   # x=0 face
        [4, 6, 7], [4, 7, 5],   # x=1 face
        [0, 4, 5], [0, 5, 1],   # y=0
        [2, 3, 7], [2, 7, 6],   # y=1
        [0, 2, 6], [0, 6, 4],   # z=0
        [1, 5, 7], [1, 7, 3],   # z=1
    ])
    assert mesh_volume_m3(v, f) == pytest.approx(1.0)


def _report() -> dict:
    return {
        "meta": {"seed": 0, "num_bodies": 2},
        "bodies": [
            {"values": {"G07": 69.0, "A01": 160.0}, "skipped": {}},
            {"values": {"G07": 75.5, "A01": 172.2}, "skipped": {}},
        ],
    }


def test_compare_reports_identical_passes() -> None:
    assert compare_reports(_report(), _report()) == []


def test_compare_reports_flags_drift_and_tolerates_noise() -> None:
    cur = copy.deepcopy(_report())
    cur["bodies"][1]["values"]["G07"] = 75.54  # within 0.05 tol
    assert compare_reports(_report(), cur) == []
    cur["bodies"][1]["values"]["G07"] = 75.8
    fails = compare_reports(_report(), cur)
    assert len(fails) == 1 and "G07" in fails[0] and "body 1" in fails[0]


def test_compare_reports_flags_disappeared_and_new_codes() -> None:
    cur = copy.deepcopy(_report())
    del cur["bodies"][0]["values"]["A01"]
    cur["bodies"][0]["skipped"]["A01"] = "landmark failed"
    cur["bodies"][1]["values"]["ZZ9"] = 1.0
    fails = compare_reports(_report(), cur)
    assert any("A01 disappeared" in f and "landmark failed" in f
               for f in fails)
    assert any("ZZ9 newly appeared" in f for f in fails)


def test_compare_reports_meta_mismatch_fails_fast() -> None:
    cur = copy.deepcopy(_report())
    cur["meta"]["seed"] = 99
    fails = compare_reports(_report(), cur)
    assert len(fails) == 1 and "meta mismatch" in fails[0]


@pytest.mark.skipif(
    not MODEL_FILE.is_file() or MODEL_FILE.stat().st_size < 1_000_000,
    reason="SMPL-X female model file not present",
)
@pytest.mark.skipif(
    not SNAPSHOT.is_file(),
    reason="synthetic snapshot not generated yet "
           "(scripts/validate_synthetic.py --write)",
)
def test_synthetic_snapshot_regression() -> None:
    """Full gate: re-extract the sampled bodies, compare to the snapshot."""
    from tailor_twin.measure.synthetic import (
        load_snapshot,
        run_harness,
    )

    baseline = load_snapshot(SNAPSHOT)
    meta = baseline["meta"]
    current = run_harness(
        model_folder=str(REPO_ROOT / "data" / "body_models"),
        gender=meta["gender"],
        num_bodies=meta["num_bodies"],
        active_betas=meta["active_betas"],
        num_betas=meta["num_betas"],
        seed=meta["seed"],
        apose_deg=meta["apose_deg"],
        progress=False,
    )
    failures = compare_reports(baseline, current)
    if failures:
        pytest.fail(
            f"{len(failures)} synthetic-body drift(s):\n  "
            + "\n  ".join(failures[:40]))
