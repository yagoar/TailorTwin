"""Sapiens2 pose gate for the capture webapp.

Each shot the phone takes is POSTed to the server and run through
Sapiens2 308-keypoint pose *before* it is accepted. The gate enforces
the two things the downstream silhouette / pointmap fit silently
depends on:

* **front** — the subject squarely faces the camera (shoulders wide and
  level, hips level). A yawed "front" biases every width.
* **side**  — the subject is turned a true 90°. The side photo is the
  *only* source of body depth; a 10° error there foreshortens the
  depth and is the single largest measurement error in the pipeline.

Turn angle is read from the shoulder keypoints: apparent shoulder
separation ≈ true_width · cos(yaw). Wide separation → facing front;
collapsed → 90° profile. Hips corroborate. The leg keypoints are
ignored — dark clothing defeats them and they are not needed here.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

# COCO body keypoint indices inside the Goliath-308 set.
KP_SHOULDER_L, KP_SHOULDER_R = 5, 6
KP_HIP_L, KP_HIP_R = 11, 12

# Shoulder-separation / torso-length ratio of a true frontal pose — used
# to turn the measured ratio into a yaw angle.
FRONT_REF_RATIO = 0.34
# Gate thresholds (ratio = keypoint x-separation / torso length).
FRONT_MIN_SHOULDER = 0.26    # below this the "front" pose is too yawed
SIDE_MAX_SHOULDER = 0.14     # above this the "side" turn is short of 90°
SIDE_MAX_HIP = 0.20
LEVEL_TOL = 0.07             # |Δy| / torso for "shoulders/hips level"
MIN_SCORE = 0.5              # keypoint confidence floor


def _yaw_deg(shoulder_ratio: float) -> float:
    """Body yaw from frontal (0°) to profile (90°)."""
    import math
    c = min(max(shoulder_ratio / FRONT_REF_RATIO, 0.0), 1.0)
    return math.degrees(math.acos(c))


def check_pose(jpeg_path: Path, view: str,
               *, model_size: str = "0.4b") -> dict:
    """Run Sapiens pose on one shot and verdict it for ``view``.

    Returns ``{ok, view, yaw_deg, issues: [...], detected: bool}``.
    ``ok`` is True only when the pose is good enough to accept.
    """
    from ..fit.pointmap import run_pose

    jpeg_path = Path(jpeg_path)
    with tempfile.TemporaryDirectory() as td:
        in_dir = Path(td) / "in"
        out_dir = Path(td) / "out"
        in_dir.mkdir()
        shutil.copy(jpeg_path, in_dir / "shot.jpg")
        js = run_pose(in_dir, out_dir, model_size=model_size)
        data = json.loads(js.read_text())

    frames = data.get("frames", [])
    inst = frames[0].get("instances", []) if frames else []
    if not inst:
        return {"ok": False, "view": view, "yaw_deg": None,
                "detected": False,
                "issues": ["no person detected — step into frame"]}

    kp = inst[0]["keypoints"]
    sc = inst[0]["keypoint_scores"]

    def pt(i):
        return kp[i], sc[i]

    (slx, sly), ssl = pt(KP_SHOULDER_L)
    (srx, sry), ssr = pt(KP_SHOULDER_R)
    (hlx, hly), shl = pt(KP_HIP_L)
    (hrx, hry), shr = pt(KP_HIP_R)

    issues: list[str] = []
    if min(ssl, ssr) < MIN_SCORE:
        issues.append("shoulders unclear — better lighting / contrast")

    torso = abs((sly + sry) / 2 - (hly + hry) / 2)
    if torso < 1.0:
        return {"ok": False, "view": view, "yaw_deg": None,
                "detected": True,
                "issues": ["pose unclear — stand straight, fill the frame"]}

    sh_ratio = abs(slx - srx) / torso
    hip_ratio = abs(hlx - hrx) / torso
    sh_tilt = abs(sly - sry) / torso
    hip_tilt = abs(hly - hry) / torso
    yaw = _yaw_deg(sh_ratio)

    if view in ("front", "back"):
        if sh_ratio < FRONT_MIN_SHOULDER:
            issues.append(f"not square to camera (yaw ≈ {yaw:.0f}°) — "
                          "face the lens straight on")
        if sh_tilt > LEVEL_TOL:
            issues.append("shoulders not level — stand relaxed, even")
        if hip_tilt > LEVEL_TOL:
            issues.append("hips not level — weight on both feet")
    else:  # side
        if sh_ratio > SIDE_MAX_SHOULDER or hip_ratio > SIDE_MAX_HIP:
            short = 90.0 - yaw
            issues.append(f"not a full 90° turn (≈ {yaw:.0f}°) — "
                          f"turn {short:.0f}° further")

    return {"ok": not issues, "view": view, "yaw_deg": round(yaw, 1),
            "detected": True, "issues": issues}
