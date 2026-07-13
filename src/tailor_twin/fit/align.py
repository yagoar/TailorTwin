"""Coarse rigid alignment between scan and SMPL-X canonical mesh.

Scan is assumed to be in SMPL-X-compatible coord frame:
    +X anatomical left, +Y up, +Z anterior.
This is verified visually before calling. If the assumption fails,
the chamfer loss will diverge — heatmap inspection (see viz.py) will
catch it.
"""
from __future__ import annotations

import numpy as np


def initial_transl(
    scan_verts: np.ndarray,
    smplx_canonical_verts: np.ndarray,
) -> np.ndarray:
    """Translation that aligns vertex centroids."""
    return scan_verts.mean(0) - smplx_canonical_verts.mean(0)
