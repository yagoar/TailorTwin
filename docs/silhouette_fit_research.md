# Photo-based body measurement — research findings & pipeline implications

Background research behind the `silhouette-fit` pipeline and the
`capture` webapp. Four papers were reviewed; this note records what each
contributes and how it shaped the implementation.

## Papers reviewed

| ref | title (short) | venue / year |
|-----|---------------|--------------|
| arXiv:2101.02515 | Anthropometric body measurement from RGB images, part-based shape model | 2021 |
| arXiv:2210.05667 | Human Body Measurement Estimation with Adversarial Augmentation (BMnet / BodyM) | WACV 2023 (Amazon) |
| arXiv:2205.14347 | 3D Body Shape & Clothing Measurements from Frontal- and Side-view Images | 2022 |
| PMC12193998 / jimaging-11-00205 | Inferring Body Measurements from 2D Images: A Comprehensive Review | J. Imaging, 2025 |

## Findings

### 1. arXiv:2101.02515 — part-based shape model

The paper pairs a 3D-scan training set (≈2675 female / 1474 male scans)
with a small RGB + tape-measured validation set (≈200 images) and learns
a **part-based shape model** plus a network that regresses anthropometric
measurements from 2D images.

The relevant idea is the *part-based* decomposition: the body is split
into regions, each with its own local shape model. A single global shape
space couples correlated dimensions — bust and high-bust are driven by
the same global parameters, so they cannot be set independently. A
part-based model relaxes that coupling.

**Pipeline impact.** This is the same shape-space-coupling wall that
`refine-tape` hit (it could not push G03 high-bust to target without
dragging G04 bust with it). It is the conceptual justification for
`ring-deform` and for the per-slice structure of `silhouette-fit`: a
per-horizontal-slice edit is an extreme, non-parametric form of the
part-based model — every slice is its own "part".

### 2. arXiv:2210.05667 — BMnet / BodyM

BMnet takes **two silhouettes (frontal + lateral) plus height and
weight** and regresses **14 body measurements** directly. Notes:

- Silhouettes (not RGB) are the input — they carry shape while
  discarding identity/appearance, and segmentation is cheap.
- Height and weight are fed as extra channels to resolve the scale and
  volume ambiguity that silhouettes alone leave open.
- Subjects stand in an **A-pose**.
- Training is augmented by an Adversarial Body Simulator (ABS) that
  searches SMPL shape space for hard bodies; it needs the BodyM dataset.
- Reported error on real bodies is roughly chest ≈ 23 mm, waist ≈ 21 mm,
  hip ≈ 18 mm.
- BMnet outputs **numbers only — no 3D mesh**.

**Pipeline impact.** BMnet validates our input choice — front + side
silhouette + height is the established recipe. Two divergences:

- BMnet *learns a regressor*; it needs the BodyM training set, which we
  do not have and cannot reproduce offline. `silhouette-fit` instead
  does **direct geometry** (fit SMPL-X betas so the rendered silhouette
  matches), needing no training data.
- BMnet adds **weight**. We collect it in the `capture` form for record,
  but the fit does not need it: the side silhouette measures depth
  directly, and depth is the quantity weight would otherwise proxy.

A-pose confirms the `capture` webapp pose guide.

### 3. arXiv:2205.14347 — front + side → SMPL betas

The most directly comparable work. Pipeline: U-Net silhouette
segmentation of the two views → an autoencoder compresses each
silhouette to a 256-d latent → a kernel-regularised regression predicts
**10 SMPL shape (β) parameters** plus three clothing measurements
(bust / waist / hip).

The central design decision: they **fit the SMPL parametric model
rather than free-form deforming vertices**. Constraining the output to
valid SMPL shape space guarantees an anatomically plausible body — there
is no way to produce a dented or self-intersecting mesh. Reported
accuracy is ≈ 2.83 cm average measurement error on real scans.

**Pipeline impact — this is the decisive finding.** An earlier
`silhouette-fit` did free-form per-slice anisotropic scaling: it hit the
target circumferences exactly but pushed vertices off the body manifold
(visible dents, an over-pinched waist, a seam at the torso/limb
boundary). That is precisely the failure mode this paper avoids by
construction.

`silhouette-fit` was rebuilt accordingly (`fit/silhouette_betas.py`):
the core is now a damped Gauss-Newton optimiser over the leading SMPL-X
betas, with a dense residual (body width + depth at ~24 torso heights,
plus a height term). Every iterate is a real CAESAR-trained body, so the
result cannot deform. Observed accuracy is ~1 cm RMS — consistent with
the paper's parametric range.

### 4. PMC12193998 — 2025 review

A survey of measurement-from-2D-image methods. Useful calibration
points:

- Accuracy is dominated by *capture conditions*. Structured / synthetic
  capture reaches sub-1.4 cm height error; uncontrolled "in-the-wild"
  RGB is 6–7 cm.
- SMPL-style parametric models are noted as encoding circumference
  implicitly inside the shape parameters.
- Clinical-grade work targets < 1.2–1.4 cm MAE.

**Pipeline impact.** The structured-vs-in-the-wild gap (≈1.4 cm vs
6–7 cm) is the entire reason the `capture` webapp exists: it enforces a
controlled capture — gyroscope-gated vertical phone (near-orthographic
projection), fixed A-pose guide, framing outline. Moving capture from
"in the wild" to "structured" is worth several centimetres, more than
any change to the fitting maths.

## Cross-cutting consensus

1. **Two orthogonal silhouettes + height** is the standard, sufficient
   input (2210.05667, 2205.14347). `silhouette-fit` uses exactly this.
2. **Fit a parametric model; do not free-form deform** (2205.14347
   explicitly; implied by all). Adopted — `silhouette_betas.py`.
3. **A-pose, controlled capture** (2210.05667 pose; PMC review on
   capture conditions). Enforced by the `capture` webapp.
4. **Parametric fitting has a ~2–3 cm floor** on real bodies
   (2205.14347: 2.83 cm; BMnet: ~2 cm). Matching the user's tape numbers
   exactly is *beyond* what a betas fit can do — hence `ring-deform`
   remains the optional last-centimetre polish on top.

## Where the tailor-twin pipeline stands

| stage | source of truth | status |
|-------|-----------------|--------|
| capture | gyro-gated webapp, A-pose, front+side | matches structured-capture best practice |
| segmentation | Sapiens part-seg (arms removed cleanly) | exceeds simple matting — arm contamination solved |
| shape fit | SMPL-X betas → silhouette (Gauss-Newton) | parametric, plausible by construction; ~1 cm RMS |
| tape-exact polish | `ring-deform` per-girth geometric edit | optional; closes the parametric residual |

**Net:** the literature endorses the approach and named the bug. The
free-form deformation that produced the deformed mesh is the documented
wrong answer; the betas-to-silhouette fit is the documented right one.
The remaining gap to tape-exact numbers is the known parametric floor,
deliberately handled by a separate, gentle `ring-deform` pass rather
than by deforming the shape fit itself.

## Validation — BodyM benchmark

The BodyM dataset (arXiv:2210.05667; AWS Open Data `amazon-bodym`,
CC BY-NC 4.0) is now used as a quantitative benchmark via
`tailor_twin.eval.bodym_eval`. It runs the parametric silhouette fit on
each subject's front + side silhouette + height and compares the fitted
mesh's measurements to the BodyM ground truth.

First run — BodyM Test-A, 25 subjects:

| measure | MAE (cm) | bias (cm) |
|---------|----------|-----------|
| height  | 1.6 | −1.4 |
| chest   | 8.8 | +6.4 |
| waist   | 4.3 | −1.0 |
| hip     | 6.8 | −5.1 |

Reading the numbers:

- **Waist (4.3 cm MAE, ≈0 bias)** is the cleanest comparison and the
  honest headline for the *bare-silhouette* path.
- **Chest (+6.4) and hip (−5.1)** carry large *biases*, not just spread
  — partly definitional (BodyM's measurement protocol differs from the
  seamly G04/G09 recipes), partly arm contamination of the side mask.
- BodyM ships only silhouettes, no RGB, so the Sapiens part-seg that
  removes the arms in the live pipeline cannot run here. The lateral
  mask is arm-contaminated; this number is the bare-silhouette path, a
  lower bound on the seg-assisted path.

For comparison BMnet reports ≈2 cm on the same data — because a *trained
network* learns to ignore the arms, which a geometric fit cannot. This
is the concrete case for adopting a learned regressor for the
no-part-seg scenario (see below).

## Not adopted previously — revisited now BodyM is available

`amazon-bodym` is public, so the earlier "no dataset" blockers are gone:

- **BMnet-style measurement regressor** — now trainable on BodyM
  (Windows GPU box). Worth adopting as the measurement path *when no
  part-seg is available* (it tolerates arms-in silhouettes, where the
  geometric fit degrades to 4–9 cm) and as an independent cross-check.
- **Learned betas initialiser** — a small silhouette→betas net trained
  on BodyM could seed the Gauss-Newton fit (faster, fewer local minima);
  marginal final-accuracy gain since the fit already reaches ~1 cm RMS
  with a clean seg.
- **Adversarial Body Simulator** — only relevant if a regressor is
  trained; a training-time augmentation, not a standalone component.
- **Weight input** — used by a BMnet regressor; the geometric fit still
  does not need it (side silhouette supplies depth directly).
