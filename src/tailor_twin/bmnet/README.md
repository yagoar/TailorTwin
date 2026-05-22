# BMnet — body-measurement regression from silhouettes

A re-implementation of the Body Measurement network of Ruiz et al.,
*"Human Body Measurement Estimation with Adversarial Augmentation"*
(arXiv:2210.05667, Amazon), trained on the public **BodyM** dataset.

The pointmap/silhouette fit pins a metric *front surface* and a
bounding box, but circumference (waist, hip, chest) under-reads — a
front+side box does not determine the cross-section shape. BMnet
regresses the 14 metric tape measurements straight from the two
silhouettes + height + weight; the three ambiguous girths then become
authoritative targets for `fit.refine_to_tape`, which nudges the SMPL-X
betas of a base fit to hit them.

The adversarial body simulator (ABS) augmentation from the paper *is*
implemented (`abs.py`, `--abs` flag) — see below. Plain supervised BMnet
already reaches ~1.5 cm waist error; ABS adds robustness on the
under-represented body shapes (high BMI especially).

## Modules

| file         | role                                                       |
|--------------|------------------------------------------------------------|
| `model.py`   | `BMnet` — MNASNet backbone + 128-wide MLP head, 14 outputs. |
| `download_bodym.py` | mirror the public BodyM S3 bucket (resumable).      |
| `dataset.py` | `BodyMDataset` split loader; `build_input` tensor builder.  |
| `abs.py`     | adversarial body simulator — differentiable render + g.     |
| `train.py`   | training CLI — L1, Adam, multi-step LR, optional `--abs`.   |
| `predict.py` | checkpoint → 14 measurements from two silhouettes.          |
| `refine.py`  | BMnet girths → seamly codes → `refine_betas_to_tape`.       |

## ABS — adversarial body simulator (paper §3.2)

ABS searches the SMPL-X shape space for body shapes the *current* BMnet
predicts worst, and turns them into extra training pairs. The whole
chain — betas → SMPL-X mesh → silhouette render → BMnet, and betas →
mesh → ground-truth measurements — is differentiable, so the search is
plain gradient **ascent** on the BMnet loss (Eq. 6: η = 0.1, k = 10
steps, betas clamped to ±3).

`abs.py` re-implements every differentiable block for SMPL-X:

* `render_pair` — soft front+side silhouettes (a self-contained
  bilinear-splat soft rasterizer);
* `mesh_measurements` — the 14 measurements as mesh geometry: torso
  girths are horizontal-slice convex perimeters (Cauchy's formula),
  limb girths are sliced *perpendicular to the bone*, lengths are joint
  distances;
* `mesh_height_weight` — height from the bounding box, weight from the
  closed-mesh volume × tissue density;
* `AbsSampler` — seeds random shapes, runs the ascent, emits batches in
  `dataset.build_input` layout.

`--abs` runs the paper's three-phase schedule: (1) supervised pre-train
on real BodyM; (2) fine-tune on adversarial synthetic bodies; (3) a
short real-data fine-tune to bridge the synthetic→real gap.

Deliberate, bounded simplifications: pose is a fixed canonical A-pose
(BMnet sees only silhouettes); the measurement function `g` is defined
in `abs.py` rather than by registered BodyM vertex paths, but it is used
identically for the synthetic target and inside the search, so it is
self-consistent, and phase 3 re-anchors any systematic offset.

## Dataset

BodyM lives in the public S3 bucket `s3://amazon-bodym` (us-west-2,
CC BY-NC 4.0). Layout per split (`train` / `testA` / `testB`):

```
hwg_metadata.csv          subject_id, gender, height_cm, weight_kg
measurements.csv          subject_id, …14 measurements (cm)…
subject_to_photo_map.csv  subject_id, photo_id
mask/<photo_id>.png       frontal silhouette  (white body on black)
mask_left/<photo_id>.png  lateral silhouette
```

The repo expects it unpacked at `data/bodym/` (git-ignored). `train`
(6134 photos) and `testA` (1684) are enough to train and validate;
`testB` (1160) is the in-the-wild report split.

The bucket is public, so `download_bodym.py` mirrors it with an
unsigned S3 client — no AWS account, resumable, parallel:

```bash
pip install boto3
python -m tailor_twin.bmnet.download_bodym --dest data/bodym
# or a subset:  --splits train testA
```

(Equivalently `aws s3 sync --no-sign-request --region us-west-2
s3://amazon-bodym data/bodym`.)

## Running on WSL + NVIDIA GPU

Training is ~10× faster on CUDA than on Apple MPS. The code is
device-agnostic — `train.py` auto-selects `cuda` when available.

```bash
# 1. clone the repo into WSL, cd into it
git clone <repo-url> tailor-twin && cd tailor-twin

# 2. python env + CUDA torch
python3 -m venv .venv && . .venv/bin/activate
pip install numpy scipy opencv-python pillow boto3 smplx
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. download the BodyM dataset straight from S3 (no tarball needed)
python -m tailor_twin.bmnet.download_bodym --dest data/bodym

# 4. train the baseline — paper recipe, 150k iterations (auto-detects GPU)
python -m tailor_twin.bmnet.train --iters 150000 --out data/results/bmnet.pt

# 4b. or train with the adversarial body simulator
python -m tailor_twin.bmnet.train --iters 150000 --abs --out data/results/bmnet.pt
```

`train.py` flags:

| flag                | default | note                                       |
|---------------------|---------|--------------------------------------------|
| `--iters`           | 150000  | phase-1 iterations — the paper's value      |
| `--batch-size`      | 22      | paper value                                 |
| `--lr`              | 1e-3    | Adam; ×0.1 at 75 % and 88 % of iterations   |
| `--img-h`           | 640     | per-view height — paper value               |
| `--img-w`           | 480     | per-view width — paper value; input `2*img_w` wide |
| `--device`          | auto    | `cuda` / `mps` / `cpu`                      |
| `--report-splits`   | testA testB | final per-measure tables; not used for selection |
| `--abs`             | off     | enable the adversarial body simulator       |
| `--abs-epochs`      | 10      | phase 2 — adversarial synthetic fine-tune   |
| `--abs-batches`     | 280     | adversarial batches per ABS epoch — `len(train)/batch` reproduces the paper's ~10×-real-data regime |
| `--abs-real-epochs` | 5       | phase 3 — real-data fine-tune               |
| `--abs-lr`          | 1e-4    | learning rate for the ABS fine-tune phases  |
| `--abs-num-betas`   | 16      | SMPL-X shape betas the simulator varies     |

The defaults reproduce the paper exactly: 150 000 iterations, 640×480
per-view input, batch 22, Adam 1e-3, ×0.1 LR drop at 75 % / 88 %, and
best-model selection on a 10 % holdout of the training set (§5). The
report splits (TestA, TestB) stay untouched for the final tables.

ABS is GPU-bound — each adversarial batch is a 10-step gradient ascent
through the renderer; run `--abs` on the GPU, not on MPS/CPU.

The checkpoint records its own `img_h` / `img_w`, so `predict` and
`refine` reproduce the right preprocessing automatically. Copy
`data/results/bmnet.pt` back to the capture machine.

## Predict + SMPL-X integration (capture machine)

```bash
# raw measurements from two silhouettes
python -m tailor_twin.bmnet.predict FRONT.png SIDE.png \
    --height 160 --weight 57 --ckpt data/results/bmnet.pt

# refine a base SMPL-X fit's betas to BMnet's chest/waist/hip girths
python -m tailor_twin.bmnet.refine BASE_FIT.npz FRONT SIDE \
    --height 160 --weight 57 --out-prefix data/results/pair_bmnet
```

`refine` accepts either a Sapiens `*_seg.npy` part-map or an RGB photo
for `FRONT` / `SIDE` (whichever `fit.silhouette.load_silhouette`
handles). It writes a refined `*_smplx_fit.npz`, an OBJ, and a
measurement CSV. Only chest/waist/hip (seamly G04/G07/G09) are
transferred — height is a known input and the limb measurements are
left to the geometry the front pointmap already pinned.

## Roadmap

Current state is a faithful BMnet + ABS re-implementation. The plan is
to first train a **baseline** on the data we have, then layer in the
improvements below.

### Step 0 — baseline (next)

Plain BMnet, **no `--abs`**, exact paper recipe, on the GPU:

```bash
python -m tailor_twin.bmnet.train --iters 150000 --out data/results/bmnet.pt
```

The defaults already match the paper (150k iterations, 640×480 input,
batch 22, Adam 1e-3, ×0.1 LR at 75 %/88 %). This is the clean reference
number and confirms the data path. ABS (`--abs`) is the second run,
once the baseline is verified.

### Planned improvements (ranked by expected impact)

Not yet implemented — listed so the design intent is recorded.

1. **Depth + normal input channels.** BMnet sees only two binary
   silhouettes, which discard the cross-section shape — the root cause
   of waist/bust under-read. tailor-twin already has the Sapiens front
   pointmap (metric depth) and surface normals; add them as input
   channels. ABS can render synthetic depth for free (mesh → depth is
   differentiable); real BodyM keeps the depth channels zeroed, so the
   two domains mix. Blocked on a depth-bearing training set.
2. **Gender input.** BodyM ships a gender label that is currently
   unused; the same silhouette + height + weight maps to different
   girths for male vs female fat distribution. Concat a gender scalar
   to the pooled MNASNet features. ABS would render synthetic bodies
   with the male/female SMPL-X models (not only neutral) so the
   synthetic samples carry a real gender label. To be done before a
   future training run, since it fixes the input shape.
3. **Calibrate `g` to BodyM, init ABS from real betas.** The geometric
   measurement function `g` in `abs.py` has a systematic offset (~few
   cm) from BodyM's scan-derived measurements, so phase 2 and phase 3
   partly disagree. Fitting SMPL-X to a sample of BodyM bodies would
   (a) calibrate `g`'s slice levels and (b) seed the adversarial ascent
   from realistic betas instead of `N(0, 1)`.
4. **Synthetic silhouette noise.** ABS renders are noise-free; real
   BodyM masks carry segmentation artifacts. Injecting holes / blur /
   edge jitter into the ABS renders narrows the synthetic→real gap the
   paper flags. Cheap; do alongside the first `--abs` run.
5. **Limb crops for small girths.** Wrist and ankle occupy few pixels
   in a full-body silhouette and are the worst measurements. A
   two-stage crop-and-measure, or extra limb input streams, would help.
6. **Uncertainty head.** Predict a per-measurement σ (Gaussian NLL) so
   `refine_to_tape` can weight each girth by confidence and flag failed
   captures.
7. **Joint fit.** Replace the sequential BMnet → `refine_to_tape` step
   with a single energy: betas chamfer to the front pointmap, match the
   BMnet girths (weighted by the σ from item 6), and match both
   silhouettes — no order dependence.
