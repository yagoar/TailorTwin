# TailorTwin

Personal 3D body measurement tool for sewing pattern drafting.

Capture yourself with an iPhone Pro LiDAR sensor, fit a parametric body
model (SMPL-X+D) to the scan, extract the 160+ measurements that the
**Aldrich** (5th ed.) and **dresspatternmaking.com** systems need to
draft bodice / sleeve / skirt / pants blocks, and view everything in a
browser-based 3D viewer with measurement overlays.

This is a single-user, local-only project. No cloud, no auth, no
shared backend.

---

## Quickstart

```bash
git clone <repo> tailor-twin && cd tailor-twin
python -m venv .venv && source .venv/bin/activate
pip install -e .
# Drop SMPL-X model file into data/body_models/smplx/SMPLX_FEMALE.npz
# (get it from https://smpl-x.is.tue.mpg.de/)
tailor-twin gui                    # opens http://127.0.0.1:8060/
```

Fill in the form, pick a Stray Scanner capture folder, hit **Run scan**.
First-time runs take ~3-6 min on Apple Silicon.

## CLI

After `pip install -e .` the `tailor-twin` console script is on PATH:

| Command | Purpose |
|---|---|
| `tailor-twin gui` | Web GUI (Flask + Three.js viewer) |
| `tailor-twin scan CAPTURE --out-prefix data/results/NAME` | End-to-end pipeline |
| `tailor-twin preflight CAPTURE` | Pre-scan capture sanity check |
| `tailor-twin bent-arm FIT_NPZ` | Re-pose elbow + dump L01/L02/L04 |
| `tailor-twin ring-deform FIT_NPZ --target G04=88 …` | Tape-exact girths on an existing fit |
| `tailor-twin refine-tape FIT_NPZ …` | Betas-solve alternative to ring-deform |
| `tailor-twin <cmd> --help` | Full flag list per command |

Direct module invocation also works: `python -m tailor_twin.scan …`,
`python -m tailor_twin.measure.cli …`.

### Calibrated scan (recommended)

The bare fit lands within a few cm on each girth. Pass a tape-measured
height and any girths you care about to pin them exactly:

```bash
tailor-twin scan data/captures/NAME --out-prefix data/results/NAME \
  --use-displacement --waist-color cyan \
  --height 160 \
  --tape-anchor G04=87.5 --tape-anchor G07=69 \
  --tape-anchor G09=99 --tape-anchor M07=34
```

Anchorable codes: torso `G03` high-bust, `G04` bust, `G05` underbust,
`G07` waist, `G08` high-hip, `G09` hip; legs `M03` thigh, `M05` knee,
`M07` calf, `M09` ankle (each leg scaled independently). The GUI's
**Calibration** card exposes the same fields. Segmentation defaults to
`rvm` (person matting); `--pose-graph` adds drift-corrected fusion.

## Pipeline

```
Stray capture       →  preprocess  →  TSDF fuse  →  cleanup
   (rgb + depth        (RVM person    (Open3D       (floor crop,
    + confidence        matting +      ScalableTSDF, largest comp,
    + odometry)         depth filter)  intrinsics    smooth,
                                       rescaled)     decimate)
   ↓
SMPL-X+D fit  →  clean-fit    →  tape anchors  →  measure   →  exports
 (300 betas +   (symmetrise,    (--height +      (167 Seamly  (CSV, OBJ,
  per-vertex     head/hand mask, ring-scale       catalog +    SMIS, JSON)
  displacement)  A-pose)         girths exact)    bent-arm)
```

The fit is parametric (~1-3 cm vs tape, repeatable). The optional
`--height` and `--tape-anchor` pass then pin the scale and any tape-
measured girths *exactly*, while the scan keeps the real cross-section
shape — validated end-to-end against tape (height/bust/waist/hip/calf
within ~0.1 cm after anchoring).

All artefacts written next to the user-chosen `--out-prefix`:

```
data/results/yaiza_20260517_scan.obj          # cleaned TSDF mesh
data/results/yaiza_20260517_smplx_fit.npz     # fit parameters
data/results/yaiza_20260517_fit_body.obj      # fitted body mesh
data/results/yaiza_20260517_measurements.csv  # Seamly catalog
data/results/yaiza_20260517_aldrich.csv       # filtered named CSV
data/results/yaiza_20260517_seamly_catalog.json
data/results/yaiza_20260517.smis              # SeamlyMe XML
data/results/yaiza_20260517_bent_arm.{json,npz}
data/results/yaiza_20260517_waist_y.json      # if elastic detected
```

## Package layout

```
src/tailor_twin/
  cli.py            # Typer console script entry
  scan.py           # full pipeline runner (was scripts/run_scan.py)
  preflight.py      # capture pre-flight inspector
  io/               # Stray Scanner frame loader
  preprocess/       # depth filter, segmentation, waist-string detect
  reconstruct/      # TSDF fuse (intrinsics rescale) + mesh cleanup (floor crop)
  fit/              # SMPL-X+D fitter, clean-fit, ring-deform tape anchors
  measure/          # landmarks, Seamly catalog + extractor, exports
  gui/              # Flask app + Three.js viewer
scripts/            # dev utilities (regenerate docs, SeamlyMe export, landmark editor)
tests/              # unit + snapshot regression
docs/               # measurement catalog, recipes glossary
references/         # source PDFs/notes for Aldrich + dpm
```

## Docs

- [`SPEC.md`](SPEC.md) — project spec (pipeline, accuracy targets, schema)
- [`GUARDRAILS.md`](GUARDRAILS.md) — AI-generation rules for this repo
- [`docs/recipes.md`](docs/recipes.md) — measurement-recipe glossary
- [`docs/catalog_coverage.md`](docs/catalog_coverage.md) — auto-generated table of every Seamly code + status
- [`src/tailor_twin/measure/README.md`](src/tailor_twin/measure/README.md) — measure subpackage overview
