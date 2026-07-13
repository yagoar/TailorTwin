"""Recipe primitives for the Seamly catalog extractor.

Per references/seamly/extraction_audit.md section "Implementation sketch":
the Phase 6 generic extractor maps Seamly catalog code -> recipe primitive,
where each primitive is one of:

    Height(landmark)            -> vertical distance floor -> landmark
    PlanarGirth(plane)          -> convex-hull perimeter at horizontal plane
    PlanarArc(plane, a, b, side)-> chord-clipped arc on the plane slice
    LateralChord(plane)         -> widest X chord at the plane
    LandmarkChord(a, b)         -> straight-line 3D distance
    Geodesic(waypoints)         -> surface path through ordered points
    GeodesicLoop(waypoints)     -> closed surface loop through waypoints (G03)
    Formula(others, expr)       -> arithmetic over other Seamly codes

Recipes operate on (verts, faces, LandmarkSet). Output is centimetres.

The G03/G11 geodesic-vs-planar warning from extraction_audit.md is honoured
by routing those codes through Geodesic/GeodesicLoop in seamly_catalog.py,
not PlanarGirth/PlanarArc.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import potpourri3d as pp3d
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull, QhullError, cKDTree


from .landmarks import LandmarkSet
from .mesh_ops import (
    _build_loops,
    _convex_hull_perimeter,
    _loop_xz,
    _pick_largest_loop,
    _pick_loop_near_point,
    _pick_torso_loop,
    slice_mesh,
)
from .regions import region_vertex_mask


# Exception groups for narrowed `except` blocks. Splitting by failure
# mode means a genuine bug (TypeError, AttributeError, KeyError) bubbles
# up instead of being silently swallowed as NaN.
SOLVER_ERRORS = (RuntimeError, IndexError)  # potpourri3d geodesic solver
HULL_ERRORS = (QhullError, ValueError, IndexError)  # scipy ConvexHull


# ---------------------------------------------------------------------------
# Tolerance constants. Order-of-magnitude varies because the units do:
# coordinates are metres (~1.0 for body height) but tangent vectors / dot
# products can collapse to 1e-3 magnitudes around degenerate cases.
# ---------------------------------------------------------------------------

# Unit-axis normalisation guard. Below this, axis is treated as degenerate
# and the caller errors / returns NaN rather than dividing by ~zero.
EPS_AXIS_NORM = 1e-9

# 3D vector zero-length guard for chord lengths, line directions, etc.
# Looser than EPS_AXIS_NORM because vector inputs may already be the result
# of subtracting two near-coincident points.
EPS_VECTOR_NORM = 1e-6

# "Already on the requested Y plane" early-exit. End-of-arc Y vs target Y
# within this tolerance means we skip the trailing chord segment.
EPS_Y_PLANE_HIT = 1e-4

# Right-edge inclusion for t-bin selection in DiagonalSurfacePlumb: each
# bin is [t_i, t_{i+1} + EPS) so the last bin includes its right boundary.
EPS_BIN_EDGE = 1e-9


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


def _y_axis() -> np.ndarray:
    return np.array([0.0, 1.0, 0.0])


def _floor_y(verts: np.ndarray) -> float:
    """Floor is the minimum Y of the fitted mesh."""
    return float(verts[:, 1].min())


def _nearest_vertex(verts: np.ndarray, p: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(verts - p, axis=1)))


def _densify_last(arc: np.ndarray, target: np.ndarray,
                   step: float = 0.01) -> np.ndarray:
    """Append a straight chord from arc[-1] to `target`, interpolated
    every `step` metres so subsequent Gaussian smoothing has enough
    samples on the chord segment to round the kink at arc[-1]."""
    last = arc[-1]
    seg = target - last
    L = float(np.linalg.norm(seg))
    if L < EPS_VECTOR_NORM:
        return arc
    n = max(int(L / step), 2)
    ts = np.linspace(0.0, 1.0, n + 1)[1:]  # skip arc[-1] (already present)
    pts = last[None] + ts[:, None] * seg
    return np.vstack([arc, pts])


# ---------------------------------------------------------------------------
# Recipe protocol — each subclass implements compute(verts, faces, landmarks).
#
# Some recipes (Height, LandmarkChord, VerticalDrop, PolylineChord) need
# only `landmarks` to compute their value. Their `verts` / `faces` params
# are kept for protocol parity so the catalog can dispatch all recipes
# uniformly via `recipe.compute(verts, faces, landmarks)`. Static
# analysers will flag them as unused — that's expected.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Height:
    """Vertical distance from floor (min Y) to landmark, in cm."""
    landmark: str

    def compute(self, verts, faces, landmarks: LandmarkSet) -> float:
        p = landmarks[self.landmark]
        return (p[1] - _floor_y(verts)) * 100.0


@dataclass(frozen=True)
class PlanarGirth:
    """Convex-hull perimeter of the horizontal slice at the landmark's Y.

    regions: tuple of region names from regions.REGIONS. If set, only
    triangles entirely inside those regions are sliced. Default ("torso",)
    keeps arms/legs out of torso-circumference slices."""
    landmark: str
    regions: tuple[str, ...] = ("torso",)

    def _slice_loop(self, verts, faces, landmarks):
        origin = landmarks[self.landmark]
        mask = region_vertex_mask(self.regions) if self.regions else None
        segs = slice_mesh(verts, faces, origin, _y_axis(), vertex_mask=mask)
        loops = _build_loops(segs)
        if not loops:
            return None
        # For non-torso regions (e.g. left_arm), the mask isolates the body
        # part; if both knees survive (e.g. mid_knee slice across both legs),
        # pick the loop closest to the landmark's actual 3D position.
        if "torso" in self.regions:
            return _pick_torso_loop(loops)
        return _pick_loop_near_point(loops, origin)

    def compute(self, verts, faces, landmarks: LandmarkSet) -> float:
        loop = self._slice_loop(verts, faces, landmarks)
        if loop is None:
            return float("nan")
        xy = _loop_xz(loop, _y_axis())
        return _convex_hull_perimeter(xy) * 100.0


@dataclass(frozen=True)
class PlanarArc:
    """Arc on a horizontal slice, clipped between two landmarks. side='front'
    picks the higher-Z arc; side='back' the lower-Z arc."""
    landmark_plane: str
    clip_start: str
    clip_end: str
    side: str  # "front" | "back"
    regions: tuple[str, ...] = ("torso",)

    def compute(self, verts, faces, landmarks: LandmarkSet) -> float:
        origin = landmarks[self.landmark_plane]
        mask = region_vertex_mask(self.regions) if self.regions else None
        segs = slice_mesh(verts, faces, origin, _y_axis(), vertex_mask=mask)
        loops = _build_loops(segs)
        loop = _pick_torso_loop(loops) if "torso" in self.regions \
            else _pick_largest_loop(loops)
        if loop is None:
            return float("nan")
        start = landmarks[self.clip_start]
        end = landmarks[self.clip_end]
        i0 = int(np.argmin(np.linalg.norm(loop - start, axis=1)))
        i1 = int(np.argmin(np.linalg.norm(loop - end, axis=1)))
        if i0 == i1:
            return float("nan")
        n = len(loop)
        a_idx = (list(range(i0, i1 + 1)) if i0 < i1
                 else list(range(i0, n)) + list(range(0, i1 + 1)))
        b_idx = (list(range(i1, i0 + 1)) if i1 < i0
                 else list(range(i1, n)) + list(range(0, i0 + 1)))
        arc_a, arc_b = loop[a_idx], loop[b_idx]
        is_front = self.side == "front"
        if (arc_a[:, 2].mean() > arc_b[:, 2].mean()) == is_front:
            arc = arc_a
        else:
            arc = arc_b
        diffs = np.diff(arc, axis=0)
        return float(np.sqrt((diffs ** 2).sum(axis=1)).sum()) * 100.0


@dataclass(frozen=True)
class LateralChord:
    """Maximum X-extent of the horizontal slice at the landmark plane."""
    landmark: str
    regions: tuple[str, ...] = ("torso",)

    def compute(self, verts, faces, landmarks: LandmarkSet) -> float:
        origin = landmarks[self.landmark]
        mask = region_vertex_mask(self.regions) if self.regions else None
        segs = slice_mesh(verts, faces, origin, _y_axis(), vertex_mask=mask)
        loops = _build_loops(segs)
        loop = _pick_torso_loop(loops) if "torso" in self.regions \
            else _pick_largest_loop(loops)
        if loop is None:
            return float("nan")
        return float(loop[:, 0].max() - loop[:, 0].min()) * 100.0


@dataclass(frozen=True)
class LandmarkChord:
    """Straight-line 3D distance between two landmarks.

    `straight=True` opts out of the surface-drape visualisation: the
    line renders as the actual 3D chord (with a small normal offset for
    visibility) instead of being snapped to the body. Use for tape-pull
    measurements whose direction is the measurement (e.g. bustpoint to
    waist front, J04) rather than measurements where the tape lies on
    the body (e.g. H05 plumb-line on torso).
    """
    a: str
    b: str
    straight: bool = False

    def compute(self, verts, faces, landmarks: LandmarkSet) -> float:
        pa = landmarks[self.a]
        pb = landmarks[self.b]
        return float(np.linalg.norm(pb - pa)) * 100.0


@dataclass(frozen=True)
class VerticalDrop:
    """Vertical line from a landmark at its (x, z) down (or up) to the Y
    plane of another landmark, measured as a straight 3D distance.

    Used for sewing measurements where the rule is "straight down from
    X to the Y line" — the path is perpendicular to the floor, not a
    body-surface geodesic.
    """
    landmark: str
    target_y_landmark: str

    def _endpoints(self, landmarks: LandmarkSet) -> tuple[np.ndarray, np.ndarray]:
        p = landmarks[self.landmark]
        ty = landmarks[self.target_y_landmark][1]
        return p, np.array([p[0], ty, p[2]])

    def compute(self, verts, faces, landmarks: LandmarkSet) -> float:
        a, b = self._endpoints(landmarks)
        return float(abs(a[1] - b[1])) * 100.0


@dataclass(frozen=True)
class PolylineChord:
    """Sum of straight-line 3D distances between an ordered sequence of
    landmarks. Used when a tape is laid as a yardstick touching several
    anatomical points (e.g. neck-front → bust-apex → waist-front).

    `straight=True` opts out of surface-drape so the line is rendered as
    the literal chord polyline rather than a body-surface geodesic.
    """
    landmarks: tuple[str, ...]
    straight: bool = False

    def compute(self, verts, faces, landmarks: LandmarkSet) -> float:
        pts = [landmarks[n] for n in self.landmarks]
        total = 0.0
        for a, b in zip(pts, pts[1:]):
            total += float(np.linalg.norm(b - a))
        return total * 100.0


# Geodesic solvers are expensive to construct (build edge structure); cache
# per mesh CONTENT. Never key this on id(): a freed verts array's address
# gets reused by the next body's allocation, so id() collides and hands
# body k+1 the solver built on body k's mesh — geodesics silently computed
# on the wrong body (few-cm shifts in G02/G03/G10/G11 and everything
# derived; heap-layout-dependent, so it flips between processes). Bitten
# by exactly that in the synthetic harness and ring-deform audit (both
# extract several meshes per process).
_GEODESIC_CACHE: dict[bytes, pp3d.EdgeFlipGeodesicSolver] = {}
_GEODESIC_CACHE_MAX = 4  # FIFO cap; harness runs 30+ meshes per process


def _get_solver(verts: np.ndarray, faces: np.ndarray) -> pp3d.EdgeFlipGeodesicSolver:
    key = hashlib.sha1(np.ascontiguousarray(verts).tobytes()).digest() \
        + hashlib.sha1(np.ascontiguousarray(faces).tobytes()).digest()
    if key not in _GEODESIC_CACHE:
        while len(_GEODESIC_CACHE) >= _GEODESIC_CACHE_MAX:
            _GEODESIC_CACHE.pop(next(iter(_GEODESIC_CACHE)))
        _GEODESIC_CACHE[key] = pp3d.EdgeFlipGeodesicSolver(
            verts.astype(np.float64), faces.astype(np.int64)
        )
    return _GEODESIC_CACHE[key]


@dataclass(frozen=True)
class Geodesic:
    """Open geodesic path through ordered landmark waypoints (>=2).

    With `via=True` (default), intermediate waypoints are CONSTRAINTS —
    each consecutive pair (w_i, w_{i+1}) is solved as its own geodesic
    and the results concatenated, so the path is guaranteed to visit
    every waypoint. With `via=False`, the path uses
    `find_geodesic_path_poly`, which iteratively shortens the polyline
    and may skip past intermediate constraints (faster, but the
    waypoints become hints rather than visits).
    """
    waypoints: tuple[str, ...]
    via: bool = True
    smooth: bool = False  # post-smooth chained geodesic to remove V-kinks at waypoints

    def _compute_path(self, verts, faces, landmarks: LandmarkSet) -> np.ndarray | None:
        solver = _get_solver(verts, faces)
        v_ids = [_nearest_vertex(verts, landmarks[w]) for w in self.waypoints]
        if self.via:
            segments = []
            for a, b in zip(v_ids, v_ids[1:]):
                try:
                    seg = np.asarray(solver.find_geodesic_path(a, b))
                except SOLVER_ERRORS:
                    return None
                if len(seg) >= 2:
                    segments.append(seg)
            if not segments:
                return None
            out = [segments[0]]
            for s in segments[1:]:
                out.append(s[1:])
            path = np.vstack(out)
        else:
            try:
                path = np.asarray(solver.find_geodesic_path_poly(v_ids))
            except SOLVER_ERRORS:
                return None
        if self.smooth and len(path) >= 9:
            # Gaussian-smooth the polyline (endpoints pinned) to round
            # the V-kinks where chained geodesic segments meet at
            # interior waypoints. Smoothed curve is left in 3D space
            # (not re-snapped to vertices) — re-snapping produces a
            # discrete staircase artefact.
            sigma = max(2.0, len(path) * 0.06)
            sm = np.empty_like(path)
            for j in range(3):
                sm[:, j] = gaussian_filter1d(path[:, j], sigma=sigma,
                                              mode="nearest")
            sm[0] = path[0]
            sm[-1] = path[-1]
            path = sm
        return path

    def compute(self, verts, faces, landmarks: LandmarkSet) -> float:
        path = self._compute_path(verts, faces, landmarks)
        if path is None or len(path) < 2:
            return float("nan")
        diffs = np.diff(path, axis=0)
        return float(np.sqrt((diffs ** 2).sum(axis=1)).sum()) * 100.0


@dataclass(frozen=True)
class GeodesicLoop:
    """Closed geodesic loop through ordered landmark waypoints (>=3).

    We compute each pair (w_i, w_{i+1}) as an open geodesic path and
    concatenate. This guarantees the loop *visits* every waypoint, unlike
    potpourri3d's find_geodesic_loop which iteratively shortens past
    them — fine for finding the canonical shortest loop, wrong when the
    waypoints are constraints (e.g. forcing the highbust path to pass
    through a midline-back point higher than the lateral armfolds)."""
    waypoints: tuple[str, ...]

    def compute(self, verts, faces, landmarks: LandmarkSet) -> float:
        solver = _get_solver(verts, faces)
        v_ids = [_nearest_vertex(verts, landmarks[w]) for w in self.waypoints]
        v_ids.append(v_ids[0])  # close the loop
        total = 0.0
        for a, b in zip(v_ids, v_ids[1:]):
            try:
                path = solver.find_geodesic_path(a, b)
            except SOLVER_ERRORS:
                return float("nan")
            if len(path) < 2:
                continue
            diffs = np.diff(path, axis=0)
            total += float(np.sqrt((diffs ** 2).sum(axis=1)).sum())
        return total * 100.0


@dataclass(frozen=True)
class Formula:
    """Arithmetic over already-computed Seamly codes.
    expr is a Python expression; locals = {seamly_code: value_cm}."""
    expr: str

    def compute_from(self, values: dict[str, float]) -> float:
        return float(eval(self.expr, {"__builtins__": {}}, dict(values)))


@dataclass(frozen=True)
class SurfacePlumb:
    """Vertical strip on the body surface at a fixed X column.

    Starts at `start` landmark, ends at the body surface at the same X
    column at `target_y_landmark`'s Y plane. Path samples Y in steps and
    at each Y picks the front-of-body vertex at the start landmark's X
    (max Z in a small X slab). Result is a smooth curve that hugs the
    body surface but stays at constant X — a tape pulled straight down
    along the chest from the neck side, not a geodesic that curves
    laterally toward the chest centerline.
    """
    start: str
    target_y_landmark: str
    side: str = "front"  # "front" (Z>0) or "back" (Z<0)
    x_band: float = 0.015
    y_band: float = 0.006

    def _path(self, verts, faces, landmarks: LandmarkSet) -> np.ndarray | None:
        """Walk the body surface straight down from `start` to the target Y,
        with the column PINNED to the start landmark's X (sx).

        At each Y step we read the surface depth — back-most (min Z) for
        side='back', front-most (max Z) for side='front' — among vertices in
        a thin X slab around sx, but emit the point at exactly (sx, y, z).
        Pinning X is the documented "constant X column" intent: it removes
        the lateral zigzag (and the length inflation) that came from
        appending each row's extreme *vertex* at its own wandering X, which
        drifted ±x_band per step for an off-centre start (e.g. SN_L). The
        column is then oriented to begin at the start landmark, and the
        whole 3D polyline is Gaussian-smoothed (endpoints pinned) to round
        the start corner and clean per-row Z pick noise. The front/back
        split is relative to the body's median Z, so the path is
        insensitive to where the body sits in Z."""
        s = landmarks[self.start]
        sx = float(s[0])
        end_y = float(landmarks[self.target_y_landmark][1])
        y_top = max(float(s[1]), end_y)
        y_bot = min(float(s[1]), end_y)
        z_mid = float(np.median(verts[:, 2]))
        step = max(self.y_band, 1e-3)

        ys: list[float] = []
        zs: list[float] = []
        y = y_top
        while y >= y_bot - 1e-9:
            col = verts[(np.abs(verts[:, 0] - sx) < self.x_band)
                        & (np.abs(verts[:, 1] - y) < step)]
            if self.side == "back":
                col = col[col[:, 2] < z_mid]
                z = float(col[:, 2].min()) if len(col) else None
            else:
                col = col[col[:, 2] > z_mid]
                z = float(col[:, 2].max()) if len(col) else None
            if z is not None:
                ys.append(y)
                zs.append(z)
            y -= step
        if len(ys) < 2:
            return None
        cols = np.column_stack([np.full(len(ys), sx), ys, np.asarray(zs)])
        # `cols` always run top→bottom (descending Y). Orient so the walk
        # begins AT the start landmark: if `start` sits at the bottom (an
        # upward measurement, e.g. waist→lowbust), reverse to ascend from
        # it. Otherwise prepending a bottom landmark to a top→bottom list
        # folds the path into a V (down, then back up) that double-counts
        # the span.
        if abs(float(s[1]) - y_bot) < abs(float(s[1]) - y_top):
            cols = cols[::-1]
        path = np.vstack([s[None], cols])
        # Gaussian-smooth the whole 3D polyline (endpoints pinned) to round
        # the corner where the start landmark (which can sit a few cm off
        # the surface column in Z, e.g. the side-neck point set back over
        # the shoulder) meets the first surface sample, and to clean
        # per-row Z pick noise. X is already constant so it is unaffected.
        if len(path) >= 9:
            sm = np.empty_like(path)
            for j in range(3):
                sm[:, j] = gaussian_filter1d(path[:, j], sigma=2.0,
                                              mode="nearest")
            sm[0] = path[0]
            sm[-1] = path[-1]
            path = sm
        return path

    def compute(self, verts, faces, landmarks: LandmarkSet) -> float:
        path = self._path(verts, faces, landmarks)
        if path is None:
            return float("nan")
        diffs = np.diff(path, axis=0)
        return float(np.linalg.norm(diffs, axis=1).sum()) * 100.0


@dataclass(frozen=True)
class SurfacePlumbThenDrop:
    """SurfacePlumb from `start` down to `mold_y_landmark.Y` on the body
    surface, then a STRAIGHT chord from that surface endpoint to either:
      - a body landmark (`end_landmark` set), or
      - a vertical drop at the surface endpoint's X/Z to `target_y_landmark.Y`
        (default).

    Mental model: tape pinned at the neck side, hugs the chest down to
    the bust line, then continues as a straight yardstick segment to
    either a fixed landmark (e.g. waist body point at SN's X column)
    or a vertical drop down to a Y plane.
    """
    start: str
    mold_y_landmark: str
    target_y_landmark: str | None = None
    end_landmark: str | None = None
    side: str = "front"

    def _path(self, verts, faces, landmarks: LandmarkSet) -> np.ndarray | None:
        sp = SurfacePlumb(self.start, self.mold_y_landmark, side=self.side)
        upper = sp._path(verts, faces, landmarks)
        if upper is None or len(upper) < 2:
            return None
        if self.end_landmark is not None:
            target = landmarks[self.end_landmark]
            return np.vstack([upper, target[None]])
        if self.target_y_landmark is None:
            return upper
        end = upper[-1]
        ty = float(landmarks[self.target_y_landmark][1])
        if abs(end[1] - ty) < EPS_Y_PLANE_HIT:
            return upper
        drop = np.array([end[0], ty, end[2]])
        return np.vstack([upper, drop[None]])

    def compute(self, verts, faces, landmarks: LandmarkSet) -> float:
        path = self._path(verts, faces, landmarks)
        if path is None:
            return float("nan")
        return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) * 100.0


@dataclass(frozen=True)
class DiagonalSurfacePlumb:
    """Surface strip between two landmarks. Slicing plane contains the
    chord(start, end) AND is parallel to the Z axis (vertical, tilted
    to align with the front-view diagonal).

    From the front view (XY projection) the curve is the straight
    diagonal between start and end; in 3D it follows the body contour
    in Z. Generalisation of SurfacePlumb (X-perpendicular plane) to
    an arbitrary XY direction.

    Optional `mold_y_landmark`: contour the body above that Y, then
    drop to `end` with a straight chord below (same idea as
    SurfacePlumbThenDrop, used for tapes that should not mold the
    waist concavity).

    Optional `chord_endpoint`: if set, the arc (after mold_y clip if
    any) is followed by a straight chord to this landmark instead of
    the default `end`. Lets the same primitive express "mold body
    along the SN→apex diagonal, then continue at that angle to
    waist_cf" (H06: chord_endpoint='waist_cf', no mold_y).

    Optional `truncate`: if True, no chord is appended after the arc
    (regardless of mold_y / chord_endpoint). Used when the measurement
    ends ON the mold line (e.g. H16: arc clipped at G03 Y, stop there).
    """
    start: str
    end: str
    side: str = "front"  # "front" = highest mean Z; "back" = lowest
    mold_y_landmark: str | None = None
    chord_endpoint: str | None = None
    truncate: bool = False
    smooth: bool = False  # Gaussian-smooth the final polyline

    def _path(self, verts, faces, landmarks: LandmarkSet) -> np.ndarray | None:
        s = landmarks[self.start]
        e = landmarks[self.end]
        d = e - s
        d_xy = np.array([float(d[0]), float(d[1]), 0.0])
        if np.linalg.norm(d_xy) < EPS_VECTOR_NORM:
            return None
        normal = np.array([d_xy[1], -d_xy[0], 0.0])
        normal /= np.linalg.norm(normal)
        segs = slice_mesh(verts, faces, s, normal, vertex_mask=None)
        if not segs:
            return None
        loops = _build_loops(segs)
        if not loops:
            return None
        # Pick the loop whose closest vertex covers BOTH endpoints.
        # _pick_loop_near_point uses centroid, which fails when the
        # slice produces multiple loops (e.g. head + torso) and the
        # chord midpoint sits near the head centroid by coincidence.
        def _score(lp: np.ndarray) -> float:
            return (float(np.linalg.norm(lp - s, axis=1).min())
                    + float(np.linalg.norm(lp - e, axis=1).min()))
        loop = min(loops, key=_score) if loops else None
        if loop is None or len(loop) < 4:
            return None
        # Parameterise loop points along the chord (s → e). t=0 at s,
        # t=1 at e. Keep only points whose t is roughly in [0, 1] AND on
        # the correct side of the body (Z sign). Then bin by t and pick
        # the most front (max Z) / most back (min Z) point per bin so the
        # path stays a single-valued curve over t (no zigzag).
        u = e - s
        u2 = float(u @ u)
        if u2 < EPS_AXIS_NORM:
            return None
        t_param = ((loop - s) @ u) / u2
        # Z sign filter — drop the opposite-side body crossings so the
        # binner can't pick a near-zero point from the wrong lobe.
        z_thresh = 0.01 if self.side == "front" else -0.01
        z_mask = (loop[:, 2] > z_thresh) if self.side == "front" \
            else (loop[:, 2] < z_thresh)
        t_mask = (t_param >= -0.02) & (t_param <= 1.02)
        mask = z_mask & t_mask
        if mask.sum() < 3:
            return None
        sel = loop[mask]
        t_sel = t_param[mask]
        nbins = 50
        tbins = np.linspace(0.0, 1.0, nbins + 1)
        out = []
        for i in range(nbins):
            in_bin = sel[(t_sel >= tbins[i])
                         & (t_sel < tbins[i + 1] + EPS_BIN_EDGE)]
            if len(in_bin) == 0:
                continue
            if self.side == "front":
                idx = int(np.argmax(in_bin[:, 2]))
            else:
                idx = int(np.argmin(in_bin[:, 2]))
            out.append(in_bin[idx])
        if len(out) < 2:
            return None
        arc = np.asarray(out)
        # Order by t ascending so the path runs s → e.
        arc_t = ((arc - s) @ u) / u2
        arc = arc[np.argsort(arc_t)]
        # Snap arc endpoints to exact s / e only when the first/last
        # binned point is genuinely far from the landmark; otherwise the
        # bin already represents the surface near the endpoint and
        # prepending the literal s creates a kink visible in the viewer.
        if np.linalg.norm(arc[0] - s) > 0.02:
            arc = np.vstack([s[None], arc])
        else:
            arc[0] = s
        if np.linalg.norm(arc[-1] - e) > 0.02:
            arc = np.vstack([arc, e[None]])
        else:
            arc[-1] = e
        # Apply mold_y clip: keep arc points strictly above (or below
        # for upward-going) the mold Y, then interpolate the arc-edge
        # to land EXACTLY at mold_y so the path ends on the requested
        # plane, not at the nearest bin Y.
        if self.mold_y_landmark is not None:
            my = float(landmarks[self.mold_y_landmark][1])
            cross = None
            for i in range(len(arc) - 1):
                y1, y2 = float(arc[i, 1]), float(arc[i + 1, 1])
                if (y1 - my) * (y2 - my) < 0:
                    t = (my - y1) / (y2 - y1)
                    cross = arc[i] + t * (arc[i + 1] - arc[i])
                    break
            if s[1] >= e[1]:
                kept = arc[arc[:, 1] >= my]
            else:
                kept = arc[arc[:, 1] <= my]
            if cross is not None and len(kept) >= 1:
                arc = np.vstack([kept, cross[None]])
            elif len(kept) >= 2:
                arc = kept
        # Decide what (if anything) to append after the arc.
        if self.truncate:
            out = arc
        elif self.chord_endpoint is not None:
            target = landmarks[self.chord_endpoint]
            out = _densify_last(arc, target)
        elif self.mold_y_landmark is not None:
            # Default: chord back to the diagonal endpoint after clipping.
            out = _densify_last(arc, e)
        else:
            out = arc
        if self.smooth and len(out) >= 9:
            sigma = max(2.0, len(out) * 0.06)
            sm = np.empty_like(out)
            for j in range(3):
                sm[:, j] = gaussian_filter1d(out[:, j], sigma=sigma,
                                              mode="nearest")
            sm[0] = out[0]
            sm[-1] = out[-1]
            out = sm
        return out

    def compute(self, verts, faces, landmarks: LandmarkSet) -> float:
        path = self._path(verts, faces, landmarks)
        if path is None:
            return float("nan")
        return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) * 100.0


@dataclass(frozen=True)
class GeodesicThenDrop:
    """Surface geodesic through ordered waypoints, then a STRAIGHT vertical
    chord from the last waypoint down to `target_y_landmark.Y` at the
    last waypoint's X/Z.

    Used for tapes that follow the body curve down to one level (e.g.
    waist→low_hip along the side) and then drop straight to a target
    Y plane (e.g. the floor).
    """
    waypoints: tuple[str, ...]
    target_y_landmark: str

    def _path(self, verts, faces, landmarks: LandmarkSet) -> np.ndarray | None:
        solver = _get_solver(verts, faces)
        v_ids = [_nearest_vertex(verts, landmarks[w]) for w in self.waypoints]
        try:
            geo = np.asarray(solver.find_geodesic_path_poly(v_ids))
        except SOLVER_ERRORS:
            return None
        if len(geo) < 2:
            return None
        end = geo[-1]
        ty = float(landmarks[self.target_y_landmark][1])
        if abs(end[1] - ty) < EPS_Y_PLANE_HIT:
            return geo
        drop = np.array([end[0], ty, end[2]])
        return np.vstack([geo, drop[None]])

    def compute(self, verts, faces, landmarks: LandmarkSet) -> float:
        path = self._path(verts, faces, landmarks)
        if path is None:
            return float("nan")
        return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) * 100.0


@dataclass(frozen=True)
class LimbGirth:
    """Circumference perpendicular to a limb axis at a given landmark.

    The slice plane normal is `axis_to - axis_from` (e.g. elbow joint
    minus shoulder joint for the upper arm), so the cut is square to the
    limb tube rather than parallel to the floor.

    Plane origin = `landmark`. Convex-hull perimeter of the planar slice
    inside the region mask. `axis_from` / `axis_to` accept either a
    landmark name or `joint.<NAME>` (resolved via SMPL-X joints).
    """
    landmark: str
    axis_from: str
    axis_to: str
    regions: tuple[str, ...] = ()
    radius_m: float | None = None  # extra mask: keep only verts within radius
    # of `landmark` (3D distance). Useful when the region mask alone is too
    # broad — e.g. bent-arm elbow slice where upper arm + forearm both lie
    # inside the `left_arm` region but only the elbow cross-section is wanted.

    def _plane(self, landmarks: LandmarkSet) -> tuple[np.ndarray, np.ndarray]:
        a = landmarks[self.axis_from]
        b = landmarks[self.axis_to]
        n = b - a
        norm = float(np.linalg.norm(n))
        if norm < EPS_AXIS_NORM:
            raise ValueError(f"LimbGirth({self.landmark}): degenerate axis")
        return landmarks[self.landmark], n / norm

    def _slice_loop(self, verts, faces, landmarks: LandmarkSet) -> np.ndarray | None:
        origin, normal = self._plane(landmarks)
        mask = region_vertex_mask(self.regions) if self.regions else None
        if self.radius_m is not None:
            radial = np.linalg.norm(verts - origin, axis=1) < self.radius_m
            mask = radial if mask is None else (mask & radial)
        segs = slice_mesh(verts, faces, origin, normal, vertex_mask=mask)
        if not segs:
            return None
        loops = _build_loops(segs)
        if loops:
            if len(loops) == 1:
                return loops[0]
            near = _pick_loop_near_point(loops, origin)
            return near if near is not None else _pick_largest_loop(loops)
        # No closed loop survived the region mask (the limb is fused to
        # the torso at this slice — typical at the armpit). Fall back to
        # collecting all segment points and taking their 2D convex hull
        # in the slice plane, restricted to points within radius of the
        # origin so we don't sweep in another limb.
        pts = np.vstack(segs)
        d = np.linalg.norm(pts - origin, axis=1)
        radius_m = self.radius_m if self.radius_m is not None else 0.20
        near_pts = pts[d < radius_m]
        if len(near_pts) < 3:
            return None
        return near_pts

    def compute(self, verts, faces, landmarks: LandmarkSet) -> float:
        loop = self._slice_loop(verts, faces, landmarks)
        if loop is None or len(loop) < 3:
            return float("nan")
        # Convex hull perimeter in the plane: project loop to 2D using two
        # in-plane basis vectors, then ConvexHull.area in 2D == perimeter
        # for 2D ConvexHull.
        origin, normal = self._plane(landmarks)
        # Build orthonormal basis u, v ⟂ normal.
        ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = ref - (ref @ normal) * normal
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        pts2d = np.stack([(loop - origin) @ u, (loop - origin) @ v], axis=1)
        try:
            hull = ConvexHull(pts2d)
        except HULL_ERRORS:
            return float("nan")
        verts_h = pts2d[hull.vertices]
        diffs = np.diff(np.vstack([verts_h, verts_h[:1]]), axis=0)
        return float(np.linalg.norm(diffs, axis=1).sum()) * 100.0


@dataclass(frozen=True)
class TapeLoop:
    """Closed loop that simulates a tape measure pulled taut across the
    chest at highbust level. Two halves:

    - BACK: planar slice at plane_y, convex-hulled from one side
      endpoint around the CB to the other. Flat at underarm Y, body-
      contour through the back, no scapular-gutter dip.
    - FRONT: by default the planar slice at plane_y on the front side,
      convex-hulled R -> L to bridge the CF cleavage. If
      `front_waypoints` is given, the front instead follows a GEODESIC
      path through those waypoints between the side endpoints — used to
      reuse an existing curve like G11 (armfold-to-armfold geodesic
      over the bust) as the front of the highbust loop.

    Args:
      plane_landmark: name of a landmark whose Y defines the underarm-
        height slice plane (e.g. "underarm_left").
      side_endpoints: (left, right) landmark names that anchor both arcs.
        For G03 these are ("underarm_left", "underarm_right").
      front_waypoints: optional ordered landmark names visited by the
        front geodesic between side_endpoints[0] -> ... ->
        side_endpoints[1]. If None, the front is the planar-slice half-
        hull.
      regions: region mask for the planar back slice (default torso).
    """
    plane_landmark: str
    side_endpoints: tuple[str, str]
    front_waypoints: tuple[str, ...] | None = None
    regions: tuple[str, ...] = ("torso",)

    def _planar_loop(self, verts, faces, landmarks: LandmarkSet) -> np.ndarray | None:
        """Closed planar torso loop at plane_y.

        Slicing is restricted to triangles inside `regions` (default
        torso) via the SMPL-X skinning-weight vertex mask. That mask is
        fixed on the model topology, so the torso↔arm cut lands at the
        same anatomical place for every figure — independent of betas
        and pose. The arm is dropped before the slice; the armhole
        leaves a notch the downstream XZ hull bridges, exactly as a
        taut tape does across the underarm hollow."""
        plane_y = landmarks[self.plane_landmark][1]
        origin = np.array([0.0, plane_y, 0.0])
        mask = region_vertex_mask(self.regions) if self.regions else None
        segs = slice_mesh(verts, faces, origin, _y_axis(), vertex_mask=mask,
                          strict_mask=True)
        if not segs:
            return None
        loop = _pick_torso_loop(_build_loops(segs))
        if loop is not None:
            return loop
        # Dropping the arm opens the torso surface at the armholes, so
        # at underarm height no closed loop survives the stitch. The
        # half-hull consumer only needs the planar point cloud (it takes
        # a convex hull, which bridges the armhole gap exactly as a taut
        # tape does) — so fall back to the raw segment points.
        return np.vstack(segs)

    def _half_hull(self, verts, faces, landmarks: LandmarkSet,
                    side: str) -> np.ndarray | None:
        """Half of the XZ-convex hull of the planar loop at plane_y.
        side="back" -> Z<midZ half, traversed L -> R through CB.
        side="front" -> Z>midZ half, traversed R -> L through CF.
        Convex-hulling drops CF/CB concavities the way a taut tape
        would. The arm is already excluded by the torso vertex mask in
        _planar_loop, so the loop is torso-only — no geometric x-clip
        (a clip keyed off the underarm landmark would trim real torso
        on a wide figure).
        """
        loop = self._planar_loop(verts, faces, landmarks)
        if loop is None or len(loop) < 6:
            return None
        L = landmarks[self.side_endpoints[0]]
        R = landmarks[self.side_endpoints[1]]
        try:
            hidx = ConvexHull(loop[:, [0, 2]]).vertices  # CCW
        except HULL_ERRORS:
            return None
        hull = loop[hidx]
        d_L = np.linalg.norm(hull[:, [0, 2]] - L[[0, 2]], axis=1)
        d_R = np.linalg.norm(hull[:, [0, 2]] - R[[0, 2]], axis=1)
        pos_L = int(np.argmin(d_L))
        pos_R = int(np.argmin(d_R))
        if pos_L == pos_R:
            return None
        h = len(hull)
        # Two candidate arcs between L and R.
        if pos_L < pos_R:
            arc1 = list(range(pos_L, pos_R + 1))
            arc2 = list(range(pos_R, h)) + list(range(0, pos_L + 1))
        else:
            arc1 = list(range(pos_R, pos_L + 1))
            arc2 = list(range(pos_L, h)) + list(range(0, pos_R + 1))
        a1 = hull[arc1]
        a2 = hull[arc2]
        want_back = side == "back"
        if want_back:
            chosen = a1 if a1[:, 2].mean() < a2[:, 2].mean() else a2
        else:
            chosen = a1 if a1[:, 2].mean() > a2[:, 2].mean() else a2
        # Orient endpoints. Back: L -> R; Front: R -> L.
        first, last = (L, R) if want_back else (R, L)
        if np.linalg.norm(chosen[0, [0, 2]] - first[[0, 2]]) > \
           np.linalg.norm(chosen[-1, [0, 2]] - first[[0, 2]]):
            chosen = chosen[::-1]
        return np.vstack([first[None], chosen[1:-1], last[None]])

    def _back_arc(self, verts, faces, landmarks: LandmarkSet) -> np.ndarray | None:
        return self._half_hull(verts, faces, landmarks, "back")

    def _front_arch(self, verts, faces, landmarks: LandmarkSet) -> np.ndarray | None:
        if self.front_waypoints is None:
            return self._half_hull(verts, faces, landmarks, "front")
        # Front = geodesic L -> waypoints -> R on the body surface.
        L = landmarks[self.side_endpoints[0]]
        R = landmarks[self.side_endpoints[1]]
        solver = _get_solver(verts, faces)
        names = (self.side_endpoints[0],
                 *self.front_waypoints,
                 self.side_endpoints[1])
        v_ids = [_nearest_vertex(verts, landmarks[n]) for n in names]
        segs = []
        for a, b in zip(v_ids, v_ids[1:]):
            try:
                p = np.asarray(solver.find_geodesic_path(a, b))
            except SOLVER_ERRORS:
                return None
            if len(p) >= 2:
                segs.append(p)
        if not segs:
            return None
        out = [segs[0]]
        for s in segs[1:]:
            out.append(s[1:])
        front = np.vstack(out)
        # Reorient: this _half_hull-style consumers expect R -> L.
        # Geodesic ran L -> R, so flip.
        front = front[::-1]
        return np.vstack([R[None], front[1:-1], L[None]])

    def compute(self, verts, faces, landmarks: LandmarkSet) -> float:
        back = self._back_arc(verts, faces, landmarks)
        front = self._front_arch(verts, faces, landmarks)
        if back is None or front is None:
            return float("nan")
        back_len = float(np.sqrt((np.diff(back, axis=0) ** 2).sum(-1)).sum())
        front_len = float(np.sqrt((np.diff(front, axis=0) ** 2).sum(-1)).sum())
        return (back_len + front_len) * 100.0

    def polyline(self, verts, faces, landmarks: LandmarkSet) -> np.ndarray | None:
        back = self._back_arc(verts, faces, landmarks)
        front = self._front_arch(verts, faces, landmarks)
        if back is None or front is None:
            return None
        # back ends at R, front starts at R -> skip front[0] duplicate.
        return np.vstack([back, front[1:]])


# Convenience union for dispatch / typing.
PrimitiveRecipe = (
    Height | PlanarGirth | PlanarArc | LateralChord
    | LandmarkChord | VerticalDrop | PolylineChord
    | Geodesic | GeodesicLoop | TapeLoop
    | LimbGirth | SurfacePlumb
    | SurfacePlumbThenDrop | GeodesicThenDrop
    | DiagonalSurfacePlumb
)


# ---------------------------------------------------------------------------
# Polyline extraction — for visualisation. Returns an (N, 3) numpy array of
# 3D points whose total length equals the recipe's measurement value.
# ---------------------------------------------------------------------------


def drape_polyline_on_body(
    poly: np.ndarray,
    body_verts: np.ndarray,
    body_normals: np.ndarray,
    faces: np.ndarray | None = None,
    samples: int = 80,
    offset_m: float = 0.008,
) -> np.ndarray:
    """Replace a chord polyline (which may pass through the body interior)
    with the corresponding surface geodesic — the shortest path along
    the body that joins the same endpoints.

    Used when the MEASUREMENT VALUE is a straight chord or vertical drop
    (mental model: yardstick distance) but the VIZ should show the tape
    laid on the body. The geodesic is naturally smooth (shortest path on
    the mesh) so we get a clean line without ad-hoc smoothing.

    If `faces` is not provided we fall back to nearest-vertex resampling
    plus Gaussian smoothing.
    """
    if len(poly) < 2:
        return poly
    tree = cKDTree(body_verts)
    # Skip drape when the chord stays in air (e.g. a vertical plumb-line
    # from a forward-projected landmark to a Y-plane point). Detected
    # by sampling the midpoint and checking distance to nearest body
    # vertex — if too large, the line is genuinely off-surface and
    # should remain as a straight chord in the viz.
    mid = poly.mean(axis=0)
    mid_dist, _ = tree.query(mid)
    if mid_dist > 0.025:  # >2.5cm from surface = treat as in-air chord
        # Apply a uniform normal offset along the chord so the line
        # sits cleanly in front of the body rather than z-fighting.
        _, idx = tree.query(poly)
        return poly + body_normals[idx] * offset_m
    if faces is not None:
        solver = _get_solver(body_verts, faces)
        segments = []
        for a, b in zip(poly, poly[1:]):
            i = int(tree.query(a)[1])
            j = int(tree.query(b)[1])
            if i == j:
                continue
            try:
                seg = np.asarray(solver.find_geodesic_path(i, j))
            except SOLVER_ERRORS:
                continue
            if len(seg) >= 2:
                segments.append(seg)
        if segments:
            full = [segments[0]]
            for s in segments[1:]:
                full.append(s[1:])
            geo = np.vstack(full)
            # Push outward 5mm along nearest-vertex normals so it sits
            # just above the surface.
            _, idx = tree.query(geo)
            return geo + body_normals[idx] * offset_m

    # Fallback: dense resample + Gaussian smooth.
    seg = np.diff(poly, axis=0)
    seg_lens = np.linalg.norm(seg, axis=1)
    total = float(seg_lens.sum())
    if total < EPS_VECTOR_NORM:
        return poly
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    ts = np.linspace(0.0, total, samples)
    sampled = np.zeros((samples, 3))
    for i, t in enumerate(ts):
        idx = int(np.searchsorted(cum, t) - 1)
        idx = max(0, min(idx, len(seg_lens) - 1))
        local = (t - cum[idx]) / max(seg_lens[idx], EPS_AXIS_NORM)
        sampled[i] = poly[idx] + local * (poly[idx + 1] - poly[idx])
    _, nearest = tree.query(sampled)
    draped = body_verts[nearest] + body_normals[nearest] * offset_m
    n = len(draped)
    if n >= 9:
        sigma = max(3.0, n * 0.18)
        smoothed = np.zeros_like(draped)
        for j in range(3):
            smoothed[:, j] = gaussian_filter1d(draped[:, j], sigma=sigma,
                                                mode="nearest")
        smoothed[0] = draped[0]
        smoothed[-1] = draped[-1]
        draped = smoothed
    return draped


# Recipe classes whose default visualisation is a straight chord that
# may pass through the body interior. The viewer / render script drape
# these onto the body surface.
DRAPE_ON_BODY_RECIPES = (LandmarkChord, VerticalDrop, PolylineChord)


def should_drape(recipe) -> bool:
    """Return True if this recipe's polyline should be draped onto the
    body surface, False if it should stay as the literal 3D chord."""
    if not isinstance(recipe, DRAPE_ON_BODY_RECIPES):
        return False
    if isinstance(recipe, LandmarkChord) and recipe.straight:
        return False
    if isinstance(recipe, PolylineChord) and recipe.straight:
        return False
    return True


def recipe_polyline(recipe, verts, faces, landmarks: LandmarkSet) -> np.ndarray | None:
    """Return the 3D polyline that visualises a recipe on the fitted mesh.

    None if the recipe is value-only (Formula) or cannot be visualised.
    """
    try:
        if isinstance(recipe, Height):
            p = landmarks[recipe.landmark]
            return np.array([[p[0], _floor_y(verts), p[2]], p])

        if isinstance(recipe, PlanarGirth):
            loop = recipe._slice_loop(verts, faces, landmarks)
            if loop is None:
                return None
            # Show the convex hull (= taut tape around the body) rather than
            # the body-surface contour. Matches the convex_hull_perimeter
            # value the recipe computes.
            xy = _loop_xz(loop, _y_axis())
            if len(xy) < 3:
                return np.vstack([loop, loop[:1]])
            try:
                hull_idx = ConvexHull(xy).vertices
            except HULL_ERRORS:
                return np.vstack([loop, loop[:1]])
            hull_loop = loop[hull_idx]
            return np.vstack([hull_loop, hull_loop[:1]])

        if isinstance(recipe, PlanarArc):
            origin = landmarks[recipe.landmark_plane]
            mask = (region_vertex_mask(recipe.regions)
                    if recipe.regions else None)
            segs = slice_mesh(verts, faces, origin, _y_axis(),
                              vertex_mask=mask)
            loops = _build_loops(segs)
            loop = (_pick_torso_loop(loops) if "torso" in recipe.regions
                    else _pick_largest_loop(loops))
            if loop is None:
                return None
            start = landmarks[recipe.clip_start]
            end = landmarks[recipe.clip_end]
            i0 = int(np.argmin(np.linalg.norm(loop - start, axis=1)))
            i1 = int(np.argmin(np.linalg.norm(loop - end, axis=1)))
            if i0 == i1:
                return None
            n = len(loop)
            a_idx = (list(range(i0, i1 + 1)) if i0 < i1
                     else list(range(i0, n)) + list(range(0, i1 + 1)))
            b_idx = (list(range(i1, i0 + 1)) if i1 < i0
                     else list(range(i1, n)) + list(range(0, i0 + 1)))
            arc_a, arc_b = loop[a_idx], loop[b_idx]
            is_front = recipe.side == "front"
            if (arc_a[:, 2].mean() > arc_b[:, 2].mean()) == is_front:
                return arc_a
            return arc_b

        if isinstance(recipe, LateralChord):
            origin = landmarks[recipe.landmark]
            mask = (region_vertex_mask(recipe.regions)
                    if recipe.regions else None)
            segs = slice_mesh(verts, faces, origin, _y_axis(),
                              vertex_mask=mask)
            loops = _build_loops(segs)
            loop = (_pick_torso_loop(loops) if "torso" in recipe.regions
                    else _pick_largest_loop(loops))
            if loop is None:
                return None
            i_min = int(np.argmin(loop[:, 0]))
            i_max = int(np.argmax(loop[:, 0]))
            return np.array([loop[i_min], loop[i_max]])

        if isinstance(recipe, LandmarkChord):
            return np.array([landmarks[recipe.a], landmarks[recipe.b]])

        if isinstance(recipe, VerticalDrop):
            a, b = recipe._endpoints(landmarks)
            return np.array([a, b])

        if isinstance(recipe, PolylineChord):
            return np.array([landmarks[n] for n in recipe.landmarks])

        if isinstance(recipe, Geodesic):
            return recipe._compute_path(verts, faces, landmarks)

        if isinstance(recipe, GeodesicLoop):
            solver = _get_solver(verts, faces)
            v_ids = [_nearest_vertex(verts, landmarks[w]) for w in recipe.waypoints]
            v_ids.append(v_ids[0])  # close the loop
            segments = []
            for a, b in zip(v_ids, v_ids[1:]):
                p = np.asarray(solver.find_geodesic_path(a, b))
                if len(p) >= 2:
                    segments.append(p)
            if not segments:
                return None
            full = [segments[0]]
            for s in segments[1:]:
                full.append(s[1:])
            return np.vstack(full)

        if isinstance(recipe, TapeLoop):
            return recipe.polyline(verts, faces, landmarks)

        if isinstance(recipe, SurfacePlumb):
            return recipe._path(verts, faces, landmarks)

        if isinstance(recipe, SurfacePlumbThenDrop):
            return recipe._path(verts, faces, landmarks)

        if isinstance(recipe, GeodesicThenDrop):
            return recipe._path(verts, faces, landmarks)

        if isinstance(recipe, DiagonalSurfacePlumb):
            return recipe._path(verts, faces, landmarks)

        if isinstance(recipe, LimbGirth):
            loop = recipe._slice_loop(verts, faces, landmarks)
            if loop is None or len(loop) < 3:
                return None
            origin, normal = recipe._plane(landmarks)
            ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 \
                else np.array([0.0, 1.0, 0.0])
            u = ref - (ref @ normal) * normal
            u /= np.linalg.norm(u)
            v = np.cross(normal, u)
            pts2d = np.stack([(loop - origin) @ u, (loop - origin) @ v], axis=1)
            try:
                hull = ConvexHull(pts2d).vertices
            except HULL_ERRORS:
                return np.vstack([loop, loop[:1]])
            return np.vstack([loop[hull], loop[hull[:1]]])
    # Outer catch-all stays broad: this is purely viz, partial polyline
    # is better than crashing the viewer. Individual sites above narrow
    # solver/hull failures; this catches anything else (KeyError
    # from a missing landmark, AttributeError, etc.).
    except Exception:
        return None
    return None
