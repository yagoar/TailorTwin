"""Regression tests for the geodesic-solver cache in measure/primitives.

The cache was keyed on ``id(verts)``: a freed verts array's address gets
reused by the next body's allocation, so ``id()`` collided and handed
body k+1 the solver built on body k's mesh — geodesic codes (G02 / G03 /
G10 / G11 and everything derived) silently computed on the WRONG body,
flipping between processes with heap layout. Content keying makes that
unrepresentable; these tests pin the content-key semantics with a stub
solver (no real pp3d solve needed, but importing primitives requires the
potpourri3d/scipy stack).
"""
from __future__ import annotations

import numpy as np
import pytest

pp3d = pytest.importorskip("potpourri3d")

from tailor_twin.measure import primitives  # noqa: E402


class _StubSolver:
    """Records construction inputs; never touches pp3d internals."""

    instances = 0

    def __init__(self, verts, faces):
        _StubSolver.instances += 1
        self.verts = verts
        self.faces = faces


@pytest.fixture()
def stub_cache(monkeypatch):
    monkeypatch.setattr(primitives.pp3d, "EdgeFlipGeodesicSolver", _StubSolver)
    monkeypatch.setattr(primitives, "_GEODESIC_CACHE", {})
    _StubSolver.instances = 0
    return primitives._GEODESIC_CACHE


def _mesh(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(12, 3)).astype(np.float32),
            np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int32))


def test_same_content_different_objects_hit_cache(stub_cache) -> None:
    v1, f = _mesh(0)
    v2 = v1.copy()  # different object, identical bytes
    assert v1 is not v2
    s1 = primitives._get_solver(v1, f)
    s2 = primitives._get_solver(v2, f)
    assert s1 is s2
    assert _StubSolver.instances == 1


def test_different_content_never_reuses_solver(stub_cache) -> None:
    # The id()-collision failure mode: a NEW body's array must get a NEW
    # solver even if it were allocated at the old array's address.
    v1, f = _mesh(0)
    s1 = primitives._get_solver(v1, f)
    v2, _ = _mesh(1)
    s2 = primitives._get_solver(v2, f)
    assert s1 is not s2
    assert _StubSolver.instances == 2
    # And the solver handed back really was built on the matching mesh.
    np.testing.assert_array_equal(np.asarray(s2.verts, dtype=np.float32),
                                  v2.astype(np.float32))


def test_cache_is_capped_fifo(stub_cache) -> None:
    f = _mesh(0)[1]
    for seed in range(primitives._GEODESIC_CACHE_MAX + 3):
        primitives._get_solver(_mesh(seed)[0], f)
    assert len(stub_cache) <= primitives._GEODESIC_CACHE_MAX
    # Oldest entries evicted, newest retained.
    newest = _mesh(primitives._GEODESIC_CACHE_MAX + 2)[0]
    before = _StubSolver.instances
    primitives._get_solver(newest, f)
    assert _StubSolver.instances == before  # still cached
