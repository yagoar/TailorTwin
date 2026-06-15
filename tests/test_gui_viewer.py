"""Smoke tests for the /viewer page + scan-listing API."""
from __future__ import annotations

from pathlib import Path

import pytest

from tailor_twin.gui.app import create_app
from tailor_twin.gui.runner import Runner
from tailor_twin.gui.viewer_data import list_scans, scan_dir


@pytest.fixture()
def app(tmp_path: Path):
    a = create_app(runner=Runner(cwd=tmp_path))
    a.config["TESTING"] = True
    return a


@pytest.fixture()
def client(app):
    return app.test_client()


def test_viewer_page_renders(client) -> None:
    r = client.get("/viewer")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "3D Viewer" in body
    assert "scan-picker" in body
    assert 'class="cam-btn"' in body
    assert "viewer.js" in body
    # Camera preset 1..5 buttons present.
    for n in range(1, 6):
        assert f'data-cam="{n}"' in body


def test_list_scans_empty(tmp_path: Path) -> None:
    assert list_scans(tmp_path) == []


def test_list_scans_marks_obj(tmp_path: Path) -> None:
    (tmp_path / "alpha_smplx_fit.npz").write_bytes(b"")
    (tmp_path / "alpha_fit_body.obj").write_text("v 0 0 0\n")
    (tmp_path / "beta_smplx_fit.npz").write_bytes(b"")
    out = list_scans(tmp_path)
    by_name = {s["name"]: s for s in out}
    assert by_name["alpha"]["has_obj"] is True
    assert by_name["alpha"]["obj_url"] == "/api/scan/alpha/obj"
    assert by_name["beta"]["has_obj"] is False
    assert by_name["beta"]["obj_url"] == ""


def test_list_scans_finds_nested_runs(tmp_path: Path) -> None:
    # GUI run: artifacts grouped in a per-run subfolder.
    sub = tmp_path / "yaiza_20260615"
    sub.mkdir()
    (sub / "yaiza_20260615_smplx_fit.npz").write_bytes(b"")
    (sub / "yaiza_20260615_fit_body.obj").write_text("v 0 0 0\n")
    (sub / "yaiza_20260615_tape_smplx_fit.npz").write_bytes(b"")
    # Legacy flat run still discoverable alongside.
    (tmp_path / "old_smplx_fit.npz").write_bytes(b"")
    out = {s["name"]: s for s in list_scans(tmp_path)}
    assert out["yaiza_20260615"]["has_obj"] is True
    assert "yaiza_20260615_tape" in out  # nested tape variant found
    assert "old" in out  # flat legacy run found


def test_scan_dir_resolves_nested_and_flat(tmp_path: Path) -> None:
    sub = tmp_path / "run1"
    sub.mkdir()
    (sub / "run1_tape_smplx_fit.npz").write_bytes(b"")
    (tmp_path / "flat_smplx_fit.npz").write_bytes(b"")
    assert scan_dir(tmp_path, "run1_tape") == sub
    assert scan_dir(tmp_path, "flat") == tmp_path
    assert scan_dir(tmp_path, "missing") == tmp_path  # fallback to root


def test_index_has_viewer_link(client) -> None:
    body = client.get("/").get_data(as_text=True)
    assert "3D Viewer" in body
    assert 'href="/viewer"' in body
