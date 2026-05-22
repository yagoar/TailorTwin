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

The adversarial body simulator (ABS) augmentation from the paper is
**not** implemented — an optional ~10 % refinement. Plain supervised
BMnet already reaches ~1.5 cm waist error.

## Modules

| file         | role                                                       |
|--------------|------------------------------------------------------------|
| `model.py`   | `BMnet` — MNASNet backbone + 128-wide MLP head, 14 outputs. |
| `dataset.py` | `BodyMDataset` split loader; `build_input` tensor builder.  |
| `train.py`   | training CLI — L1, Adam, multi-step LR, TestA validation.   |
| `predict.py` | checkpoint → 14 measurements from two silhouettes.          |
| `refine.py`  | BMnet girths → seamly codes → `refine_betas_to_tape`.       |

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
(6134 photos) and `testA` (1684) are enough to train and validate.

## Running on WSL + NVIDIA GPU

Training is ~10× faster on CUDA than on Apple MPS. The code is
device-agnostic — `train.py` auto-selects `cuda` when available.

```bash
# 1. clone the repo into WSL, cd into it
git clone <repo-url> tailor-twin && cd tailor-twin

# 2. unpack the BodyM dataset (transfer data/bodym/ or the tarball)
mkdir -p data && tar xzf bodym.tar.gz -C data/

# 3. python env + CUDA torch
python3 -m venv .venv && . .venv/bin/activate
pip install numpy scipy opencv-python pillow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. train (auto-detects the GPU)
python -m tailor_twin.bmnet.train --epochs 150 --out data/results/bmnet.pt
```

`train.py` flags:

| flag           | default               | note                                  |
|----------------|-----------------------|----------------------------------------|
| `--epochs`     | 80                    | 150k paper iters ≈ 540 epochs           |
| `--batch-size` | 22                    | paper value                             |
| `--lr`         | 1e-3                  | Adam; ×0.1 at 75 % and 88 % of epochs   |
| `--img-h`      | 256                   | per-view height                         |
| `--img-w`      | 192                   | per-view width; input is `2*img_w` wide |
| `--device`     | auto                  | `cuda` / `mps` / `cpu`                  |
| `--val-split`  | testA                 |                                         |

With a GPU you can afford the paper's full-resolution input —
`--img-h 640 --img-w 480`. The checkpoint records its own `img_h` /
`img_w`, so `predict` and `refine` reproduce the right preprocessing
automatically. Copy `data/results/bmnet.pt` back to the capture machine.

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
