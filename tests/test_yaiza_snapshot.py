"""Regression snapshot: Yaiza fit -> CSV must stay byte-identical.

Gates every refactor. If a primitive changes numerics, this test breaks
and the diff shows which measurements drifted.

Baseline: `data/results/yaiza_measurements.baseline.csv`. Regenerate
intentionally by deleting the baseline and copying the new CSV in.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
FIT_NPZ = REPO_ROOT / "data" / "results" / "yaiza_smplx_fit.npz"
BASELINE_CSV = REPO_ROOT / "data" / "results" / "yaiza_measurements.baseline.csv"


@pytest.fixture(scope="module")
def regen_csv(tmp_path_factory) -> Path:
    if not FIT_NPZ.is_file():
        pytest.skip(f"fit npz missing: {FIT_NPZ}")
    if not BASELINE_CSV.is_file():
        pytest.skip(f"baseline missing: {BASELINE_CSV}")
    out = tmp_path_factory.mktemp("yaiza") / "yaiza.csv"
    cmd = [
        ".venv/bin/python", "-m", "tailor_twin.measure.cli", str(FIT_NPZ),
        "--num-betas", "300", "--save-csv", str(out),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True)
    return out


def test_yaiza_csv_byte_identical(regen_csv: Path):
    baseline = BASELINE_CSV.read_bytes()
    new = regen_csv.read_bytes()
    if baseline == new:
        return
    # Detailed diff for failure output.
    baseline_lines = baseline.decode().splitlines()
    new_lines = new.decode().splitlines()
    diffs: list[str] = []
    for i, (b, n) in enumerate(zip(baseline_lines, new_lines)):
        if b != n:
            diffs.append(f"line {i + 1}:\n  baseline: {b}\n  new:      {n}")
    if len(baseline_lines) != len(new_lines):
        diffs.append(
            f"line-count differs: baseline={len(baseline_lines)} "
            f"new={len(new_lines)}"
        )
    pytest.fail(
        "Yaiza CSV drifted from baseline:\n" + "\n".join(diffs[:30]))


def test_modules_importable():
    """Smoke-test: imports don't crash, surface obvious breakage."""
    import tailor_twin.measure.bent_arm  # noqa: F401
    import tailor_twin.measure.cli  # noqa: F401
    import tailor_twin.measure.exports  # noqa: F401
    import tailor_twin.measure.landmarks  # noqa: F401
    import tailor_twin.measure.primitives  # noqa: F401
    import tailor_twin.measure.regions  # noqa: F401
    import tailor_twin.measure.seamly_catalog  # noqa: F401
    import tailor_twin.measure.seamly_extractor  # noqa: F401
