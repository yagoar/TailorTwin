# SHAPY photo-based front-end — handoff plan

Goal: replace the iPhone LiDAR Stray Scanner capture with a photo-only
front-end powered by [SHAPY](https://github.com/muelea/shapy). Same
back-end (refine-tape, measure CLI, viewer, garment pipeline).

This doc is the picking-up-on-the-CUDA-box checklist.

---

## Why photos > LiDAR (for accessibility)

| | Stray LiDAR | SHAPY photos |
|---|---|---|
| Hardware required | iPhone Pro / Pro Max (LiDAR) | any phone with camera |
| Capture time | 25–60 s hold-still scan | 3–5 single photos |
| Capture difficulty | high (motion drift, vertical sweep, helper needed) | low (3 angles, A-pose) |
| Sensor resolution | 192 × 256 depth | 1920 × 1440 RGB |
| Raw accuracy on circumferences | ±5–10 cm | ±2–3 cm |
| After tape refine | <0.5 cm | <0.5 cm |
| External GPU needed | no (CPU TSDF fit works) | yes (CUDA for SHAPY regressor) |

For mass distribution: photo path is the strategic choice. LiDAR path
stays as an option but isn't the main UX.

---

## Status — what's already done

### In tailor-twin (Mac CPU, tested)

- `src/tailor_twin/io/shapy_loader.py` — parses SHAPY npz, converts rotation matrices to axis-angle, fuses betas across multiple views (mean), pads CAESAR's 10 betas to whatever `num_betas` the rest of the pipeline uses.
- `src/tailor_twin/io/shapy_import_cli.py` — CLI implementation.
- `src/tailor_twin/cli.py` — registers `tailor-twin shapy-import`.
- Smoke-tested with SHAPY's bundled `samples/shapy_fit_for_virtual_measurements/img_00.npz` — produces a valid tailor-twin fit npz; `measure cli` extracts 168 measurements correctly.

### Not yet done (needs CUDA box)

- Install SHAPY itself, download trained models.
- Run SHAPY regressor on real photos of the user.
- End-to-end test: photos → SHAPY → tailor-twin → measurements vs tape.
- Tune multi-view fusion (mean betas works as baseline; may want confidence weighting later).

---

## Step-by-step on WSL Ubuntu + NVIDIA GPU

### 1. SHAPY install

```bash
# In WSL Ubuntu
cd ~  # or wherever
git clone https://github.com/muelea/shapy.git
cd shapy
export PYTHONPATH=$PYTHONPATH:$(pwd)/attributes/

python3.8 -m venv .venv/shapy
source .venv/shapy/bin/activate
pip install -r requirements.txt

cd attributes && python setup.py install && cd ..

# mesh-mesh-intersection is CUDA-only — required by some SHAPY paths
cd mesh-mesh-intersection
export CUDA_SAMPLES_INC=$(pwd)/include
pip install -r requirements.txt
python setup.py install
cd ..
```

Check CUDA recognised:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Should print `True ...` followed by your GPU name.

### 2. Download model data

Two pieces:

#### SMPL-X body model
Register at https://smpl-x.is.tue.mpg.de, download SMPL-X v1.1. Drop into:

```
shapy/data/body_models/smplx/SMPLX_NEUTRAL.npz
shapy/data/body_models/smplx/SMPLX_FEMALE.npz
shapy/data/body_models/smplx/SMPLX_MALE.npz
```

(You already have these in `tailor-twin/data/body_models/smplx/` — can symlink.)

#### SHAPY trained weights
Register at https://shapy.is.tue.mpg.de, download `shapy_data.zip`. Extract into `shapy/data/`:

```
shapy/data/expose_release/...
shapy/data/trained_models/shapy/SHAPY_A/...
shapy/data/trained_models/b2a/...
shapy/data/trained_models/a2b/...
shapy/data/utility_files/...
```

Confirm by running their demo on bundled images:

```bash
cd regressor
python demo.py --save-vis true --save-params true --save-mesh true \
  --split test --datasets openpose \
  --output-folder ../samples/shapy_fit/ \
  --exp-cfg configs/b2a_expose_hrnet_demo.yaml \
  --exp-opts output_folder=../data/trained_models/shapy/SHAPY_A \
             part_key=pose \
             datasets.pose.openpose.data_folder=../samples \
             datasets.pose.openpose.img_folder=images \
             datasets.pose.openpose.keyp_folder=openpose \
             datasets.batch_size=1 datasets.pose_shape_ratio=1.0
```

Should produce `samples/shapy_fit/<some-image>/...{vis,mesh,params}.{png,ply,npz}`.

### 3. Take your photos

Setup:
- Tight clothing (compression / underwear) — same protocol as LiDAR scan
- A-pose: arms ~30° below T-pose horizontal, palms forward
- Even diffuse light, plain non-reflective background, plain floor
- Camera at chest height, ~2 m away, full body in frame
- Phone in landscape, locked exposure if possible

Shots (4 minimum, 6 better):

| # | yaw | reason |
|---|---|---|
| 1 | 0° front | torso width, bust, hip width, shoulder-to-hip ratio |
| 2 | 90° right side | torso depth, butt curve, neck profile |
| 3 | 180° back | shoulder blades, back width, calf shape |
| 4 | 270° left side | side asymmetry vs shot 2 |
| 5 | 45° front-right | shoulder-hip diagonal |
| 6 | 315° front-left | left-side diagonal |

Save as `me_01.jpg`, `me_02.jpg`, … in `shapy/samples/images/`.

### 4. Generate OpenPose keypoints

SHAPY regressor needs 2D keypoints per image, in OpenPose JSON format.
Options:

#### Option A — OpenPose

```bash
# Heavy install (~5 GB compiled). Follow OpenPose README.
./build/examples/openpose/openpose.bin \
  --image_dir ~/shapy/samples/images \
  --write_json ~/shapy/samples/openpose \
  --display 0 --render_pose 0
```

Produces `me_01_keypoints.json`, … in `samples/openpose/`.

#### Option B — MediaPipe converter (lighter, no OpenPose build)

```bash
pip install mediapipe opencv-python
```

Write a quick converter (~50 lines) — MediaPipe Pose gives 33 landmarks,
OpenPose BODY_25 has 25 with different ordering. Map MP indices to OP
positions; emit OpenPose JSON schema. There are public gists for this
mapping — search "mediapipe to openpose body_25 json".

Save as `~/shapy/scripts/mp_to_openpose.py` for reuse.

### 5. Run SHAPY regressor on your photos

```bash
cd ~/shapy/regressor
python demo.py --save-params true --save-mesh true \
  --split test --datasets openpose \
  --output-folder ../samples/me_fit/ \
  --exp-cfg configs/b2a_expose_hrnet_demo.yaml \
  --exp-opts output_folder=../data/trained_models/shapy/SHAPY_A \
             part_key=pose \
             datasets.pose.openpose.data_folder=../samples \
             datasets.pose.openpose.img_folder=images \
             datasets.pose.openpose.keyp_folder=openpose \
             datasets.batch_size=1 datasets.pose_shape_ratio=1.0
```

Output: `samples/me_fit/me_XX/me_XX.npz` per image (+ optional mesh/vis).

### 6. Import into tailor-twin

```bash
cd ~/tailor-twin  # whatever path the repo lives at on WSL
git pull          # so the shapy-import command is present
.venv/bin/python -m pip install -e .   # if not already

tailor-twin shapy-import \
  ~/shapy/samples/me_fit/me_01/me_01.npz \
  ~/shapy/samples/me_fit/me_02/me_02.npz \
  ~/shapy/samples/me_fit/me_03/me_03.npz \
  ~/shapy/samples/me_fit/me_04/me_04.npz \
  --out-prefix data/results/me_shapy \
  --gender female --num-betas 300
```

Or point at the parent dir:

```bash
tailor-twin shapy-import \
  ~/shapy/samples/me_fit/ \
  --out-prefix data/results/me_shapy \
  --gender female --num-betas 300
```

Produces:
- `data/results/me_shapy_smplx_fit.npz` — fused fit, canonical T-pose
- `data/results/me_shapy_fit_body.obj`
- `data/results/me_shapy_measurements.csv`
- `data/results/me_shapy_seamly_catalog.json`
- `data/results/me_shapy.smis`

### 7. Refine with tape

```bash
tailor-twin refine-tape data/results/me_shapy_smplx_fit.npz \
  --out-prefix data/results/me_shapy_refined \
  --target A01=160 \
  --target G03=88 \
  --target G07=70 \
  --target G09=99 \
  --a-pose-shoulder-deg 30
```

Optionally add more tape targets:
- `--target G02=<neck>`
- `--target L11=<bicep>`
- `--target N01=<inseam>`
- `--target M01=<thigh>`
- bump `--num-betas-active 25` for ≥10 targets

### 8. Visual + diff check

```bash
tailor-twin gui
```

Pick `me_shapy_refined`. Verify:
- Pose is A-pose, not weird hand-on-hip
- HUD shows Height/Bust/Waist/Hip matching tape
- Mesh proportions look like the subject

Compare to LiDAR-derived results side-by-side:
- `data/results/scan_4b750ebd56_refined_*` — LiDAR + refine
- `data/results/me_shapy_refined_*` — photos + refine

Eyeball + numerically compare. Decide which front-end gives better
likeness for your subject.

---

## Anticipated issues + fixes

### CUDA out of memory on SHAPY
SHAPY's HRNet backbone + ExPose ~6 GB. Reduce batch size to 1 (already
default). If still OOM, run images one at a time.

### Image size mismatch
SHAPY regressor expects images at specific resolutions per config. The
`configs/b2a_expose_hrnet_demo.yaml` handles arbitrary sizes via the
ExPose crop preprocessor; if you see resolution errors, scale photos to
1080 × 1080 or similar.

### Keypoint quality
Bad keypoints = bad SHAPY output. After OpenPose / MediaPipe, sanity-
check the overlay PNG — ankle / wrist / shoulder / hip markers should
look anatomically right. Re-shoot if one image has missing or wildly
wrong keypoints.

### Multi-view divergence
If betas per view differ by >2 in any dimension (after centering),
likely:
- Different lighting between shots → SHAPY's CAESAR shape regressor
  picks up "fat looking" vs "thin looking" lighting.
- Pose drift between shots → arms higher/lower changes betas.
- Camera distance varies → SHAPY's depth-from-image cue noisy.

Mitigation: take all photos in the same session, same lighting, same
distance, same A-pose. If still divergent, the multi-view fusion
function in `shapy_loader.py` can switch to median (robust to outliers)
or trimmed mean.

### Tape refinement doesn't converge
SHAPY's 10 betas may not have enough DoF to satisfy >5 targets exactly.
Symptoms: refine-tape loops and doesn't converge.

Fix: write a "promote SHAPY betas to full-dim chamfer fit" step. After
SHAPY import, run a quick chamfer fit against the SHAPY-shaped mesh
itself (or against the LiDAR scan if available) with all 300 betas
unfrozen. This gives high-frequency shape detail SHAPY's 10 betas miss.
Code skeleton already in `src/tailor_twin/fit/fit.py` — just point at a
mesh target derived from SHAPY's `v_shaped`.

### SHAPY's BodyMeasurements vs tailor-twin's measure
SHAPY ships `body_measurements` Python package (CAESAR's measurement
standard). Your `seamly_extractor` uses Seamly/Aldrich definitions.
These differ by 1–3 cm on the same mesh because of landmark conventions
(e.g. waist circumference at narrowest section vs at navel level).

For cross-check: run SHAPY's `measurements/virtual_measurements.py` on
the same fit, compare against your CSV. Discrepancies > 2 cm point at
a landmark-definition mismatch you should document.

---

## Stretch goals (after the main path works)

### A2S tape-only path
SHAPY's `attributes/demo.py` with `04b_ahcwh2s.yaml` config: input
height + chest + waist + hip + 15 attribute ratings → SMPL-X betas.
No photos, no scan. Useful as a tape-only fallback for users without a
camera setup.

To wire into tailor-twin: same converter pattern as the regressor path
— SHAPY produces a (mesh + betas), our loader picks it up and writes
the fit_npz. Single additional CLI: `tailor-twin shapy-a2s --height
160 --chest 88 --waist 70 --hip 99 --attributes attrs.json`.

### Better keypoints than OpenPose
[ViTPose](https://github.com/ViTAE-Transformer/ViTPose) or
[MMPose RTMPose](https://github.com/open-mmlab/mmpose) give better
2D joint accuracy than OpenPose, especially on faces and hands. SHAPY
accepts any OpenPose-format JSON.

### Confidence-weighted multi-view fusion
SHAPY exposes a per-prediction confidence (`betas` are paired with
something — check `stage_n_out.keys()` in their forward output). Weight
the multi-view average by confidence instead of simple mean. Update
`shapy_loader.fuse_multi_view`.

### Drop OpenPose dependency entirely
SHAPY's repo includes the ExPose regressor; ExPose has its own
detector + cropping pipeline. The `b2a_expose_hrnet_demo.yaml` config
hides the OpenPose dependency for some image splits — investigate
whether `--datasets pose-only` or similar skips the keypoint file
requirement. If so, photo-only with zero auxiliary tools.

---

## Files / commands cheatsheet

```
src/tailor_twin/io/shapy_loader.py        # SHAPY npz parser + multi-view fuser
src/tailor_twin/io/shapy_import_cli.py    # CLI: shapy npz(s) → tailor-twin fit_npz
src/tailor_twin/cli.py                    # registers `tailor-twin shapy-import`

# Tested with:
.venv/bin/python -c "
from pathlib import Path
from tailor_twin.io.shapy_loader import load_shapy_npz, shapy_to_fit, save_fit_payload
sample = Path('/Users/ygo/Projects/Private/shapy/samples/shapy_fit_for_virtual_measurements/img_00.npz')
r = load_shapy_npz(sample)
payload = shapy_to_fit([r], out_num_betas=300, gender='female')
save_fit_payload(payload, Path('/tmp/shapy_test_fit.npz'))
"
```

```bash
# Full photo pipeline (after SHAPY installed + photos taken + keypoints generated)
tailor-twin shapy-import ~/shapy/samples/me_fit/ \
  --out-prefix data/results/me_shapy --gender female --num-betas 300

tailor-twin refine-tape data/results/me_shapy_smplx_fit.npz \
  --out-prefix data/results/me_shapy_refined \
  --target A01=160 --target G03=88 --target G07=70 --target G09=99 \
  --a-pose-shoulder-deg 30

tailor-twin gui    # pick me_shapy_refined
```

---

## Acceptance criteria

Pipeline is "working" when:
1. SHAPY regressor runs on user's photos with no errors.
2. `shapy-import` produces a valid fit npz; `measure cli` extracts ≥150 of 168 codes.
3. `refine-tape` converges within 0.5 cm on 4 targets (A01/G03/G07/G09) and ≤1 cm on extras.
4. Viewer renders the mesh in A-pose with no obvious anatomical defects.
5. CSV measurements within 2 cm of tape on every dimension that has a real tape value, before refinement; within 0.5 cm after.

If acceptance fails: log the failing dim + scan iteration, debug per
section in "Anticipated issues" above.
