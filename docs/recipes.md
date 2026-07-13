# Measurement recipes

Reference for picking the right primitive when wiring a new Seamly
code into `seamly_catalog.py::RECIPES`. All primitives live in
`src/tailor_twin/measure/primitives.py`. Every recipe is a frozen
dataclass with a `compute(verts, faces, landmarks) -> float` returning
centimetres.

## Selection cheat-sheet

| Tape mental model | Primitive |
|-------------------|-----------|
| Straight-line distance landmark→landmark | `LandmarkChord` |
| Straight 3D yardstick visiting N landmarks | `PolylineChord` |
| Vertical drop from a landmark to a Y plane | `VerticalDrop` |
| Vertical distance floor → landmark | `Height` |
| Horizontal taut tape around the torso | `PlanarGirth` |
| Front (or back) arc only of a horizontal slice | `PlanarArc` |
| Widest X-extent of a horizontal slice | `LateralChord` |
| Circumference of a limb, square to its axis | `LimbGirth` |
| Geodesic on the body through ordered points | `Geodesic` (open) / `GeodesicLoop` (closed) |
| Vertical tape clinging to body at fixed X | `SurfacePlumb` |
| As above, then straight chord to a landmark | `SurfacePlumbThenDrop` |
| Diagonal surface tape between two points, tilted vertical plane | `DiagonalSurfacePlumb` |
| Geodesic, then straight drop to floor | `GeodesicThenDrop` |
| Highbust tape: planar back arc + geodesic front | `TapeLoop` |
| Arithmetic over already-computed codes | `Formula` (see `seamly_catalog.py::FORMULAS`) |

## Recipe details

### `Height(landmark)`
Vertical floor→landmark distance. Floor = `verts[:,1].min()`.
Failure: none — always defined.

### `LandmarkChord(a, b, straight=False)`
3D euclidean distance between two landmarks. `straight=True` opts out
of surface-drape in the viz: line is rendered as the literal 3D
chord (with a small normal offset) instead of being snapped to the
body. Use for tapes pulled in space (e.g. apex→waist).

### `PolylineChord(landmarks, straight=False)`
Sum of euclidean distances between consecutive landmarks.

### `VerticalDrop(landmark, target_y_landmark)`
Vertical 3D distance: |landmark.y - target_y_landmark.y|. Value is
the Y delta; visualisation drops straight from the landmark to its
Y-projected counterpart.

### `PlanarGirth(landmark, regions=("torso",))`
Convex-hull perimeter of the horizontal slice at `landmark.y`. The
`regions` mask keeps arms / legs out of torso slices. Picks the
torso loop when multiple loops survive the slice.
Failure: empty slice (rare, e.g. floor anchor below the mesh).

### `PlanarArc(landmark_plane, clip_start, clip_end, side, regions=("torso",))`
A horizontal slice arc clipped between two landmarks on the loop.
`side="front"` picks the higher-Z (anterior) arc; `"back"` the
lower-Z. Value is arc length along the body contour.

### `LateralChord(landmark, regions=("torso",))`
Max X-extent (loop[:,0].max - .min) of the horizontal slice at the
landmark plane.

### `LimbGirth(landmark, axis_from, axis_to, regions=(), radius_m=None)`
Convex-hull perimeter of the slice perpendicular to the limb axis
`(axis_from → axis_to)` at `landmark`. `axis_from`/`axis_to` accept
landmark names or `joint.<NAME>` (e.g. `joint.L_Shoulder`).
`radius_m`: extra mask that keeps only vertices within that radius
of `landmark` — needed for bent-arm slices where region masks alone
let in opposite-side body parts.
Failure: degenerate axis (norm < EPS_AXIS_NORM) → ValueError.

### `Geodesic(waypoints, via=True, smooth=False)`
Open surface path through ordered waypoints. `via=True` chains
pairwise geodesics so every waypoint is visited (use this for
constraints); `via=False` calls `find_geodesic_path_poly` which may
skip past intermediate waypoints (faster but only valid when
waypoints are hints). `smooth=True` applies Gaussian filtering on
the resulting polyline to round V-kinks at chained-segment joins —
keep off unless the kink shows up in the viewer.
Failure: solver exception → NaN.

### `GeodesicLoop(waypoints)`
Closed loop through waypoints. Chained pairwise (always `via`),
guaranteed to visit every waypoint — unlike `find_geodesic_loop`
which iteratively shortens past them.

### `SurfacePlumb(start, target_y_landmark, side="front", x_band=0.015, y_band=0.006)`
Vertical strip on the body surface at fixed X = `start.x`. Slices
the body with a vertical plane perpendicular to X; takes the
front-half (or back) loop from Y=start.y to Y=target_y_landmark.y.
Pure planar slice — no per-Y nearest-vertex zigzag.

### `SurfacePlumbThenDrop(start, mold_y_landmark, target_y_landmark=None, end_landmark=None, side="front")`
`SurfacePlumb` down to the mold Y, then a straight chord to either
`end_landmark` (preferred) or a vertical drop at the surface
endpoint's X/Z to `target_y_landmark.y`.

### `DiagonalSurfacePlumb(start, end, side="front", mold_y_landmark=None, chord_endpoint=None, truncate=False, smooth=False)`
Surface strip whose slicing plane contains the chord(start, end)
AND is parallel to Z (vertical, tilted to align with the front-view
diagonal). Front-view projection of the curve is the straight
diagonal start→end; in 3D the curve hugs the body in Z. Optional
`mold_y_landmark`: contour the body above that Y, then straight
chord below. Optional `chord_endpoint`: end the chord at a different
landmark (e.g. waist_cf for H06). `truncate=True`: stop at the arc
(no trailing chord).

### `GeodesicThenDrop(waypoints, target_y_landmark)`
Surface geodesic through waypoints, then straight vertical chord
from the last waypoint to `target_y_landmark.y` at the last point's
X/Z. Used for outseam-style tapes (waist→side→floor).

### `TapeLoop(plane_landmark, side_endpoints, front_waypoints=None, regions=("torso",))`
Closed highbust-style loop. Back = planar slice at `plane_landmark.y`
convex-hulled from L→R via CB. Front = either the front half of the
same planar slice (if `front_waypoints is None`) or a geodesic
through `front_waypoints` between the side endpoints.

### `Formula(expr)` *(see `seamly_catalog.py::FORMULAS`)*
Arithmetic over already-computed Seamly codes. `expr` is a Python
expression evaluated with locals = `{code: value_cm}` and an empty
`__builtins__`. Use for derivations like `L03 = L01 - L02`.

## Failure modes summary

| Site | Returns | When |
|------|---------|------|
| `Geodesic*.compute` | NaN | solver raises (catches `Exception`) |
| `LimbGirth.compute` | NaN | ConvexHull degenerate / slice empty |
| `*.compute` (planar) | NaN | slice produces no loop / band < 3 pts |
| `DiagonalSurfacePlumb._path` | None → NaN | degenerate chord, t-bins all empty |
| `_get_solver` | builds new solver | re-runs are cached by `id(verts)` |

## Drape behaviour (visualisation)

`recipe_polyline(recipe, ...)` returns the 3D curve to display.
Recipes in `DRAPE_ON_BODY_RECIPES` (`LandmarkChord`, `VerticalDrop`,
`PolylineChord`) get their straight chords replaced with the geodesic
between their endpoints UNLESS `straight=True` is set on the
recipe — used when the measurement IS the yardstick line (the value
is the chord length, the viz should show the chord).
