"""BMnet — direct body-measurement regression from two silhouettes.

A re-implementation of the Body Measurement network of Ruiz et al.,
"Human Body Measurement Estimation with Adversarial Augmentation"
(arXiv:2210.05667, Amazon), trained on the public BodyM dataset.

The silhouette width/depth transfer in ``fit.silhouette_transfer`` pins a
bounding box but loses the cross-section shape, so girth (waist, hip,
chest) under-reads. BMnet sidesteps that: an MNASNet CNN regresses the
14 metric tape measurements straight from the front + side silhouettes
plus height and weight. Those girths then become authoritative targets
for ``fit.refine_to_tape``, which nudges the SMPL-X betas of a
pointmap/silhouette base fit to hit them.

Modules:
  ``model``    — BMnet (MNASNet backbone + 128-wide MLP head).
  ``dataset``  — BodyM split loader (front|side silhouette + H/W → 14 cm).
  ``train``    — training CLI (L1, Adam, multi-step LR, testA validation).
  ``predict``  — checkpoint → 14 measurements from two silhouettes.
  ``refine``   — BMnet girths → seamly codes → refine_betas_to_tape.
"""
from __future__ import annotations

# BodyM measurements.csv column order (14, height included to match the
# paper's 14-output head; height is also a network *input*).
MEAS_COLS = (
    "ankle", "arm-length", "bicep", "calf", "chest", "forearm", "height",
    "hip", "leg-length", "shoulder-breadth", "shoulder-to-crotch",
    "thigh", "waist", "wrist",
)
