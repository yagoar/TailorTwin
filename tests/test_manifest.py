"""Unit tests for the run-manifest provenance writer."""
from __future__ import annotations

import json
from pathlib import Path

from tailor_twin.manifest import git_commit, write_manifest


def test_write_manifest_roundtrip(tmp_path: Path) -> None:
    prefix = tmp_path / "run" / "yaiza_x"
    path = write_manifest(
        prefix,
        config={"capture": Path("/tmp/cap"), "height": 160.0,
                "tape_anchor": ["G04=87.5"], "native": (1920, 1440)},
        rc=0,
        started="2026-07-12T10:00:00+00:00",
        finished="2026-07-12T10:04:12+00:00",
    )
    assert path == prefix.with_name("yaiza_x_manifest.json")
    data = json.loads(path.read_text())
    assert data["exit_code"] == 0
    assert data["started"] < data["finished"]
    # Non-JSON-native config values are stringified, never dropped.
    assert data["config"]["capture"] == "/tmp/cap"
    assert data["config"]["height"] == 160.0
    # git_commit is a 40-char sha in a checkout, None outside one.
    assert data["git_commit"] is None or len(data["git_commit"]) == 40


def test_write_manifest_records_failure_rc(tmp_path: Path) -> None:
    path = write_manifest(tmp_path / "x", config={}, rc=3,
                          started="a", finished="b")
    assert json.loads(path.read_text())["exit_code"] == 3


def test_git_commit_in_this_repo() -> None:
    sha = git_commit()
    assert sha is not None and len(sha) == 40
