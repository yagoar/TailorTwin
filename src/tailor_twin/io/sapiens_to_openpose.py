"""Convert Sapiens2 308-keypoint pose predictions to OpenPose body_25
JSON files, one per image — the format SHAPY's regressor expects.

Sapiens2 keypoint ordering (first 17) follows COCO-WholeBody:

    0  nose            1  left_eye         2  right_eye
    3  left_ear        4  right_ear        5  left_shoulder
    6  right_shoulder  7  left_elbow       8  right_elbow
    9  left_wrist     10  right_wrist     11  left_hip
   12  right_hip      13  left_knee       14  right_knee
   15  left_ankle     16  right_ankle

OpenPose BODY_25 ordering:

    0  Nose            1  Neck            2  RShoulder
    3  RElbow          4  RWrist          5  LShoulder
    6  LElbow          7  LWrist          8  MidHip
    9  RHip           10  RKnee          11  RAnkle
   12  LHip           13  LKnee          14  LAnkle
   15  REye           16  LEye           17  REar
   18  LEar           19  LBigToe        20  LSmallToe
   21  LHeel          22  RBigToe        23  RSmallToe
   24  RHeel

Mapping uses direct COCO→OpenPose lookup; Neck (OP 1) and MidHip (OP 8)
are synthesised as midpoints from the relevant shoulder / hip pairs.
Foot keypoints 19-24 are left as zero — SHAPY's image-regressor demo
config only consults the body subset and tolerates zeroed foot rows.

If you later wire in a Halpe-26 / foot-keypoint-aware Sapiens2 read, the
COCO-WholeBody foot block (indices 17-22 in COCO-WholeBody) maps cleanly
into BigToe / SmallToe / Heel pairs and can be added without changing
this module's API.
"""
from __future__ import annotations

import json
from pathlib import Path


# COCO -> OpenPose BODY_25 index map. ``mid`` entries are derived from a
# pair of COCO indices (average xy, min score) — Neck and MidHip have no
# direct COCO equivalent so we synthesise them.
_OP_BODY_25 = [
    ("coco", 0),     # 0  Nose
    ("mid", 5, 6),   # 1  Neck (left+right shoulder mid)
    ("coco", 6),     # 2  RShoulder
    ("coco", 8),     # 3  RElbow
    ("coco", 10),    # 4  RWrist
    ("coco", 5),     # 5  LShoulder
    ("coco", 7),     # 6  LElbow
    ("coco", 9),     # 7  LWrist
    ("mid", 11, 12), # 8  MidHip (left+right hip mid)
    ("coco", 12),    # 9  RHip
    ("coco", 14),    # 10 RKnee
    ("coco", 16),    # 11 RAnkle
    ("coco", 11),    # 12 LHip
    ("coco", 13),    # 13 LKnee
    ("coco", 15),    # 14 LAnkle
    ("coco", 2),     # 15 REye
    ("coco", 1),     # 16 LEye
    ("coco", 4),     # 17 REar
    ("coco", 3),     # 18 LEar
    ("zero",),       # 19 LBigToe
    ("zero",),       # 20 LSmallToe
    ("zero",),       # 21 LHeel
    ("zero",),       # 22 RBigToe
    ("zero",),       # 23 RSmallToe
    ("zero",),       # 24 RHeel
]


def _instance_to_op_body_25(
    keypoints: list[list[float]],
    scores: list[float],
) -> list[float]:
    """Build the OpenPose BODY_25 flat [x,y,c, x,y,c, ...] list (75 floats)."""
    flat: list[float] = []
    for entry in _OP_BODY_25:
        kind = entry[0]
        if kind == "coco":
            idx = entry[1]
            x, y = keypoints[idx]
            c = scores[idx]
            flat.extend([float(x), float(y), float(c)])
        elif kind == "mid":
            ia, ib = entry[1], entry[2]
            xa, ya = keypoints[ia]
            xb, yb = keypoints[ib]
            ca, cb = scores[ia], scores[ib]
            flat.extend(
                [float((xa + xb) / 2.0), float((ya + yb) / 2.0),
                 float(min(ca, cb))]
            )
        elif kind == "zero":
            flat.extend([0.0, 0.0, 0.0])
        else:
            raise ValueError(f"unknown mapping kind {kind!r}")
    return flat


def _pick_primary_instance(
    instances: list[dict],
) -> dict | None:
    """Pick the most relevant person in the frame.

    Heuristic: highest mean body-keypoint score over COCO indices 5-16
    (shoulders → ankles). Falls back to bbox area if no scores."""
    if not instances:
        return None
    best = None
    best_score = -1.0
    for inst in instances:
        scores = inst.get("keypoint_scores") or []
        if len(scores) >= 17:
            body_block = scores[5:17]
            mean_score = sum(body_block) / max(1, len(body_block))
        else:
            bbox = inst.get("bbox") or [0, 0, 1, 1]
            x0, y0, x1, y1 = bbox[:4]
            mean_score = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0)) / 1e7
        if mean_score > best_score:
            best_score = mean_score
            best = inst
    return best


def convert_predictions(
    predictions_json: Path,
    out_dir: Path,
    *,
    image_height: int | None = None,
    image_width: int | None = None,
    version: str = "1.3",
) -> list[Path]:
    """Write one OpenPose-format JSON per image referenced in the
    Sapiens2 predictions JSON.

    Returns the list of output file paths so callers can confirm what
    they have to ship to SHAPY's ``--datasets pose ... keyp_folder``."""
    predictions_json = Path(predictions_json)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(predictions_json.read_text())
    if data.get("num_keypoints") != 308:
        # 17-keypoint COCO inputs would work too — only the body block
        # is consulted — but explicit num_keypoints check helps when the
        # input is misnamed (e.g. a 133-kpt prediction).
        # Don't hard-error; print and continue.
        print(f"warn: predictions reports num_keypoints="
              f"{data.get('num_keypoints')}, expected 308. "
              "Conversion will still work if first 17 follow COCO order.")

    out_paths: list[Path] = []
    for frame in data.get("frames", []):
        img_name = frame.get("image_name")
        if img_name is None:
            continue
        primary = _pick_primary_instance(frame.get("instances") or [])
        if primary is None:
            # Empty OpenPose JSON keeps the filename mapping intact for
            # downstream tools that loop over input image lists.
            op_payload = {"version": version, "people": []}
        else:
            kpts = primary.get("keypoints") or []
            scores = primary.get("keypoint_scores") or []
            if len(kpts) < 17 or len(scores) < 17:
                print(f"  skipped {img_name}: only "
                      f"{len(kpts)} kpts / {len(scores)} scores")
                op_payload = {"version": version, "people": []}
            else:
                body_25 = _instance_to_op_body_25(kpts, scores)
                op_payload = {
                    "version": version,
                    "people": [{
                        "person_id": [-1],
                        "pose_keypoints_2d": body_25,
                        "face_keypoints_2d": [],
                        "hand_left_keypoints_2d": [],
                        "hand_right_keypoints_2d": [],
                        "pose_keypoints_3d": [],
                        "face_keypoints_3d": [],
                        "hand_left_keypoints_3d": [],
                        "hand_right_keypoints_3d": [],
                    }],
                }
        stem = Path(img_name).stem
        out_file = out_dir / f"{stem}_keypoints.json"
        out_file.write_text(json.dumps(op_payload))
        out_paths.append(out_file)
    return out_paths
