# `tailor_twin.measure`

Measurement pipeline. Takes a fitted SMPL-X mesh + the Seamly catalog
of 245 sewing-pattern codes, runs each code's recipe against the
mesh, writes CSV / SMIS / JSON / OBJ artefacts.

```
.npz fit ──► extract_catalog ─► CatalogReport ─► write_csv
   │              │                      │       write_smis_from_catalog
   │              ▼                      ▼       write_obj
   │        build_landmark_set      bent_arm override
   │              │                 (L01/L02/L04 re-pose)
   │              ▼
   │        LandmarkSet (verts + vid map + joints + faces)
   │              │
   │              ▼
   └──►   RECIPES[code].compute(verts, faces, landmarks) ─► float (cm)
```

## Module map

| File | Role |
|------|------|
| `landmarks.py` | Vertex-ID + compound + dynamic landmark resolver. `LandmarkSet[name] -> (3,) float`. |
| `regions.py` | SMPL-X vertex-region masks (torso, left_arm, etc.). |
| `recipes.py` | YAML-driven recipe dispatch + mesh-slice helpers (`slice_mesh`, `_build_loops`, `_pick_torso_loop`, …). |
| `primitives.py` | Dataclass recipes: `Height`, `Geodesic`, `PlanarGirth`, `LimbGirth`, `DiagonalSurfacePlumb`, … + viz polyline. |
| `seamly_catalog.py` | The 245 Seamly catalog codes mapped to recipe instances + formulas + judgment list. |
| `seamly_extractor.py` | Run `RECIPES` over a mesh → `CatalogReport(values, skipped)`. |
| `bent_arm.py` | Re-pose SMPL-X with left elbow + shoulder flexed. Shared by CLI override and `extract_bent_arm.py`. |
| `extract_bent_arm.py` | Standalone re-pose runner; reachable as `tailor-twin bent-arm <fit.npz>` or `python -m tailor_twin.measure.extract_bent_arm`. |
| `exports.py` | CSV / OBJ / SMIS writers. |
| `cli.py` | `python -m tailor_twin.measure.cli <fit.npz> [--both] [--save-*]`. |
| `viewer.py` | Dash app for code-by-code visual review. |

## Adding a new measurement

1. Identify or add a landmark. If a single vertex suffices, append it
   to `references/smplx_landmark_review.json` (open the body in
   Blender, pick the vid, paste).
2. If the landmark is a midpoint / lerp / dynamic search, add an
   entry to `COMPOUND_LANDMARKS` or `DYNAMIC_LANDMARKS` in
   `landmarks.py`. New search types register inside `_dynamic`.
3. Pick a primitive from `primitives.py` (see `docs/recipes.md`)
   and instantiate it in `seamly_catalog.py::RECIPES` keyed by
   Seamly code.
4. Run `pytest tests/test_yaiza_snapshot.py`. It fails on byte-diff —
   if your new recipe doesn't change existing values, the test
   passes. Update the baseline (and commit it) only when the
   change is intentional.
5. Open the viewer (`python -m tailor_twin.measure.viewer`) and
   inspect the new code visually.

## Bent-arm

L01 / L02 / L04 (and the L03 formula) need an elbow-flexed mesh. The
CLI re-poses the SMPL-X body model with the L_Elbow joint rotated
~80° forward and the L_Shoulder rotated ~30° forward, then recomputes
those four codes on the bent mesh and overwrites the A-pose values.
Per-vertex displacement (SMPL-X+D) is reused unchanged.

See `bent_arm.py::repose_bent_arm` (the single source of truth) and
`docs/bent_arm.md` (TBD) for the rationale.

## Regression test

`tests/test_yaiza_snapshot.py` regenerates the Yaiza CSV and diffs
against `data/results/yaiza_measurements.baseline.csv`. Run before
and after any refactor:

```bash
.venv/bin/python -m pytest tests/test_yaiza_snapshot.py -v
```

To update the baseline intentionally:

```bash
cp data/results/yaiza_measurements.csv \
   data/results/yaiza_measurements.baseline.csv
```
