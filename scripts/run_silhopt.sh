#!/usr/bin/env bash
# Silhouette optimizer + wireframe overlay using the proven lm3 config.
#
# Hyperparams locked from the lm3 run (IoU front 0.85 / side 0.83,
# bust/waist/hip within ~2cm of tape on pair_20260522_090554). Don't
# tune these without a known-good comparison fit — defaults in the
# Python script have drifted and produce noticeably worse results.
#
# Usage:
#   scripts/run_silhopt.sh <prefix-without-suffix> [--gender female] [--height 160]
#
# Required files derived from prefix:
#   <prefix>_v3_smplx_fit.npz                  base fit (from pointmap-fit)
#   <prefix>_sapiens/out/{front,side}_seg.npy  Sapiens part segs
#   <prefix>_sapiens/pose/pose_predictions.json  Sapiens body pose
#   <prefix>_landmarks.json                    landmark editor JSON
#
# Writes:
#   <prefix>_<tag>_smplx_fit.npz
#   <prefix>_<tag>_wire_{front,side}.png
#
# Tag default = "lm" + timestamp; pass --tag X to override.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <prefix> [--tag NAME] [--gender female] [--height 160]" >&2
    exit 1
fi

PREFIX="$1"; shift
TAG="lm_$(date +%H%M%S)"
GENDER="female"
HEIGHT="160"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)    TAG="$2";    shift 2 ;;
        --gender) GENDER="$2"; shift 2 ;;
        --height) HEIGHT="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

BASE_FIT="${PREFIX}_v3_smplx_fit.npz"
FRONT_SEG="${PREFIX}_sapiens/out/front_seg.npy"
SIDE_SEG="${PREFIX}_sapiens/out/side_seg.npy"
POSE_JSON="${PREFIX}_sapiens/pose/pose_predictions.json"
LANDMARKS="${PREFIX}_landmarks.json"
OUT_PREFIX="${PREFIX}_${TAG}"

for f in "$BASE_FIT" "$FRONT_SEG" "$SIDE_SEG" "$POSE_JSON" "$LANDMARKS"; do
    [[ -f "$f" ]] || { echo "missing input: $f" >&2; exit 1; }
done

PY=".venv/bin/python"

# --- lm3 hyperparams (do not change without comparing to a known-good fit) ---
$PY scripts/silhouette_optimize.py \
    --base-fit  "$BASE_FIT" \
    --front-seg "$FRONT_SEG" \
    --side-seg  "$SIDE_SEG" \
    --landmarks "$LANDMARKS" \
    --pose-json "$POSE_JSON" \
    --out-prefix "$OUT_PREFIX" \
    --gender "$GENDER" \
    --height "$HEIGHT" \
    --num-betas 20 \
    --iters     1500 \
    --img-h     384 \
    --img-w     240 \
    --lr-betas  0.015 \
    --lr-pose   0.003 \
    --lr-cam    0.0 \
    --side-weight 1.0 \
    --tape-weight 1.0 \
    --pose-weight 1.0 \
    --gauss-sigma 0.7 \
    --pose-opt

$PY scripts/silhouette_overlay.py \
    --fit       "${OUT_PREFIX}_smplx_fit.npz" \
    --front-seg "$FRONT_SEG" \
    --side-seg  "$SIDE_SEG" \
    --out-prefix "$OUT_PREFIX" \
    --gender "$GENDER" \
    --height "$HEIGHT"

echo
echo "fit:     ${OUT_PREFIX}_smplx_fit.npz"
echo "front:   ${OUT_PREFIX}_wire_front.png"
echo "side:    ${OUT_PREFIX}_wire_side.png"
