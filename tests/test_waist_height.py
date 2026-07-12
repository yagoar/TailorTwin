"""Unit tests for the floor-relative waist-height anchor.

The waist Y override must be re-derivable in ANY frame the body ends up
in (raw scan frame, clean-fit canonical frame, ring-deformed mesh), so it
travels through the pipeline as a floor-relative height in cm and is
resolved against the mesh actually being measured.
"""
from __future__ import annotations

import numpy as np
import pytest

from tailor_twin.measure.landmarks import waist_y_from_height


def _body_like(floor_y: float, top_y: float, n: int = 50) -> np.ndarray:
    """Vertical stick of verts spanning [floor_y, top_y]."""
    ys = np.linspace(floor_y, top_y, n)
    return np.stack([np.zeros(n), ys, np.zeros(n)], axis=1)


def test_waist_y_is_floor_plus_height() -> None:
    verts = _body_like(floor_y=-1.2, top_y=0.4)
    assert waist_y_from_height(verts, 100.0) == pytest.approx(-0.2)


def test_waist_y_is_frame_invariant() -> None:
    # Same body recentred into a different frame (e.g. clean-fit canonical
    # vs ARKit world): the resolved waist stays at the same height above
    # the feet in both.
    scan_frame = _body_like(floor_y=-1.35, top_y=0.25)
    canon_frame = scan_frame.copy()
    canon_frame[:, 1] += 0.30  # recentre
    h = 98.5
    y_scan = waist_y_from_height(scan_frame, h)
    y_canon = waist_y_from_height(canon_frame, h)
    assert (y_scan - scan_frame[:, 1].min()) == pytest.approx(h / 100.0)
    assert (y_canon - y_scan) == pytest.approx(0.30)


def test_waist_height_must_be_positive() -> None:
    verts = _body_like(floor_y=0.0, top_y=1.6)
    with pytest.raises(ValueError):
        waist_y_from_height(verts, 0.0)
    with pytest.raises(ValueError):
        waist_y_from_height(verts, -5.0)
