"""Sapiens2 pointmap → metric body profile.

``silhouette.py`` measures a body outline from a 2-D mask: one global
pixel→cm scale (from the height landmark) turns the outline into width
and depth curves. Both curves inherit that single noisy scale, so the
absolute girths can drift several cm — the old fix was a tape polish.

Sapiens2's **pointmap** model predicts a metric (x, y, z) for every
pixel. Width and depth are then each measured directly in metres and
rescaled once by the known height — no shared pixel scale, no per-view
turn-angle assumption. On the project's own captures this lands chest
and hip within ~1 cm of tape with no tape input at all.

This module:

* :func:`run_sapiens` — drives the Sapiens2 ``vis_pointmap`` and
  ``vis_seg`` tools (subprocess) over a front + side photo, caching the
  ``.ply`` / ``*_seg.npy`` outputs.
* :func:`pointmap_profile` — turns one view's pointmap + part-seg into a
  :class:`~tailor_twin.fit.silhouette.SilhouetteProfile`, so the existing
  betas optimizer consumes it unchanged.

The Sapiens2 checkout and its checkpoints are located via
``$TAILOR_SAPIENS_ROOT`` / ``$SAPIENS_CHECKPOINT_ROOT`` (sensible
defaults below).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .silhouette import SilhouetteProfile

# dome29 part-seg class ids for the two arms incl. hands — dropped so
# the torso width is not inflated by an arm hanging beside it.
DOME29_ARM_CLASSES = (6, 7, 11, 15, 16, 20)
# Per-side arm. The 0.4b seg labels a *bare* arm almost entirely as the
# hand class (6 / 15) — skin reads as hand — so the per-side arm blob is
# {hand, lower-arm, upper-arm}. _axis_profile trims the end bins, so the
# included hand is dropped from the compared range anyway.
DOME29_LEFT_ARM = (6, 7, 11)
DOME29_RIGHT_ARM = (15, 16, 20)

_DEF_SAPIENS_ROOT = Path.home() / "Projects/Private/sapiens2"
_DEF_CKPT_ROOT = Path.home() / "sapiens2_host"


def _device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def _sapiens_root() -> Path:
    root = Path(os.environ.get("TAILOR_SAPIENS_ROOT", _DEF_SAPIENS_ROOT))
    if not (root / "sapiens/dense/tools/vis/vis_pointmap.py").exists():
        raise SystemExit(
            f"Sapiens2 checkout not found at {root}. Set $TAILOR_SAPIENS_ROOT "
            "to the sapiens2 repo root.")
    return root


def _ckpt_root() -> Path:
    return Path(os.environ.get("SAPIENS_CHECKPOINT_ROOT", _DEF_CKPT_ROOT))


def run_sapiens(
    photos: dict,
    work_dir: Path,
    *,
    model_size: str = "0.4b",
    seg_size: str | None = None,
    device: str | None = None,
) -> dict:
    """Run Sapiens2 pointmap + part-seg on a set of view photos.

    ``photos`` maps a view name (``"front"``, ``"side"``, ``"back"`` …)
    to its RGB photo path. ``model_size`` sizes the pointmap model;
    ``seg_size`` sizes the part-seg model (defaults to ``model_size``).

    Copies the photos into ``work_dir/in`` (the tools batch a directory),
    runs both models, and returns a dict of the per-view artifact paths.
    Inference is skipped when every view's ``.ply`` and ``_seg.npy``
    already exist, so re-runs are cheap.
    """
    seg_size = seg_size or model_size
    root = _sapiens_root()
    ckpt_root = _ckpt_root()
    device = device or _device()
    dense = root / "sapiens/dense"

    work_dir = Path(work_dir).resolve()       # subprocess runs cwd=dense
    in_dir = work_dir / "in"
    out_dir = work_dir / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    names = list(photos)
    for name in names:
        src = Path(photos[name])
        dst = in_dir / f"{name}{src.suffix.lower()}"
        for stale in in_dir.glob(f"{name}.*"):
            if stale != dst:
                stale.unlink()
        shutil.copy(src, dst)

    art = {}
    for name in names:
        photo = next(p for p in in_dir.glob(f"{name}.*"))
        art[name] = {
            "photo": photo,
            "ply": out_dir / f"{name}.ply",
            "depth": out_dir / f"{name}_depth.npy",
            "seg": out_dir / f"{name}_seg.npy",
        }

    cached = all(art[n]["ply"].exists() and art[n]["seg"].exists()
                 for n in names)
    if cached:
        print(f"sapiens: cached artifacts in {out_dir}")
        return art

    # vis_pointmap skips inference when *_depth.npy exists but only then
    # writes the .ply — drop a stale depth-without-ply so both regenerate.
    for n in names:
        if art[n]["depth"].exists() and not art[n]["ply"].exists():
            art[n]["depth"].unlink()

    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["SAPIENS_CHECKPOINT_ROOT"] = str(ckpt_root)

    pm_cfg = (dense / f"configs/pointmap/render_people/"
              f"sapiens2_{model_size}_pointmap_render_people-1024x768.py")
    pm_ckpt = (ckpt_root / f"pointmap/sapiens2-pointmap-{model_size}/"
               f"sapiens2_{model_size}_pointmap.safetensors")
    seg_cfg = (dense / f"configs/seg/shutterstock_goliath/"
               f"sapiens2_{seg_size}_seg_shutterstock_goliath-1024x768.py")
    seg_ckpt = (ckpt_root / f"seg/sapiens2-seg-{seg_size}/"
                f"sapiens2_{seg_size}_seg.safetensors")
    for f in (pm_cfg, pm_ckpt, seg_cfg, seg_ckpt):
        if not f.exists():
            raise SystemExit(f"Sapiens2 file missing: {f}")

    runs = [
        ("pointmap", model_size, "tools/vis/vis_pointmap.py", pm_cfg,
         pm_ckpt, []),
        ("seg", seg_size, "tools/vis/vis_seg.py", seg_cfg, seg_ckpt,
         ["--save_pred"]),
    ]
    for label, size, tool, cfg, ckpt, extra in runs:
        print(f"sapiens: running {label} ({size}) on {device} …")
        cmd = [sys.executable, tool, str(cfg), str(ckpt),
               "--input", str(in_dir), "--output", str(out_dir),
               "--device", device, *extra]
        r = subprocess.run(cmd, cwd=dense, env=env)
        if r.returncode != 0:
            raise SystemExit(f"sapiens {label} failed (exit {r.returncode})")

    for n in names:
        for k in ("ply", "seg"):
            if not art[n][k].exists():
                raise SystemExit(f"sapiens produced no {k} for {n}")
    return art


def run_pose(
    in_dir: Path,
    out_dir: Path,
    *,
    model_size: str = "0.4b",
    device: str | None = None,
) -> Path:
    """Run Sapiens2 308-keypoint pose on every image in ``in_dir``.

    Returns the path to the predictions JSON (one entry per image).
    Used by the capture pose-gate to verify front-squareness and a true
    90° side turn before a shot is accepted.
    """
    root = _sapiens_root()
    ckpt_root = _ckpt_root()
    device = device or _device()
    pose_dir = root / "sapiens/pose"
    in_dir = Path(in_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = (pose_dir / f"configs/keypoints308/shutterstock_goliath_3po/"
           f"sapiens2_{model_size}_keypoints308_"
           f"shutterstock_goliath_3po-1024x768.py")
    ckpt = (ckpt_root / f"pose/sapiens2-pose-{model_size}/"
            f"sapiens2_{model_size}_pose.safetensors")
    detector = ckpt_root / "detector/detr-resnet-101-dc5"
    for f in (cfg, ckpt, detector):
        if not f.exists():
            raise SystemExit(f"Sapiens2 pose file missing: {f}")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["SAPIENS_CHECKPOINT_ROOT"] = str(ckpt_root)
    cmd = [sys.executable, "tools/vis/vis_pose.py", str(detector),
           str(cfg), str(ckpt), "--input", str(in_dir),
           "--output", str(out_dir), "--device", device]
    r = subprocess.run(cmd, cwd=pose_dir, env=env)
    if r.returncode != 0:
        raise SystemExit(f"sapiens pose failed (exit {r.returncode})")
    js = sorted(out_dir.glob("*_predictions.json"))
    if not js:
        raise SystemExit("sapiens pose produced no predictions JSON")
    return js[-1]


def run_normal(
    in_dir: Path,
    out_dir: Path,
    seg_dir: Path,
    *,
    model_size: str = "0.4b",
    device: str | None = None,
) -> dict:
    """Run Sapiens2 surface-normal estimation on every image in ``in_dir``.

    ``seg_dir`` must hold the matching ``<name>_seg.npy`` part-seg masks
    (vis_normal masks the normal map to the body). Writes ``<name>.npy``
    (per-pixel unit normal, camera frame) and ``<name>.jpg`` (RGB-encoded
    normal map) into ``out_dir``. Returns ``{stem: {"npy": .., "vis": ..}}``.
    Inference is skipped when every ``.npy`` already exists.
    """
    root = _sapiens_root()
    ckpt_root = _ckpt_root()
    device = device or _device()
    dense = root / "sapiens/dense"
    in_dir = Path(in_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in in_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    art = {s: {"npy": out_dir / f"{s}.npy", "vis": out_dir / f"{s}.jpg"}
           for s in stems}
    if stems and all(a["npy"].exists() for a in art.values()):
        print(f"sapiens normal: cached in {out_dir}")
        return art

    cfg = (dense / "configs/normal/metasim_render_people/"
           f"sapiens2_{model_size}_normal_metasim_render_people-1024x768.py")
    ckpt = (ckpt_root / f"normal/sapiens2-normal-{model_size}/"
            f"sapiens2_{model_size}_normal.safetensors")
    for f in (cfg, ckpt):
        if not f.exists():
            raise SystemExit(f"Sapiens2 normal file missing: {f}")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["SAPIENS_CHECKPOINT_ROOT"] = str(ckpt_root)
    cmd = [sys.executable, "tools/vis/vis_normal.py", str(cfg), str(ckpt),
           "--input", str(in_dir), "--output", str(out_dir),
           "--seg_dir", str(Path(seg_dir).resolve()), "--device", device]
    print(f"sapiens: running normal ({model_size}) on {device} …")
    r = subprocess.run(cmd, cwd=dense, env=env)
    if r.returncode != 0:
        raise SystemExit(f"sapiens normal failed (exit {r.returncode})")
    return art


def _load_pointmap(ply: Path, h: int, w: int) -> np.ndarray:
    """Read vis_pointmap's PLY back into an (H, W, 3) metric array.

    vis_pointmap writes ``pointmap[mask].reshape(-1, 3)`` with an all-True
    mask, then appends a 500-point origin sphere — so the first H·W points
    are the row-major pointmap."""
    import open3d as o3d
    pts = np.asarray(o3d.io.read_point_cloud(str(ply)).points)
    return pts[:h * w].reshape(h, w, 3)


def load_pose_keypoints(json_path: Path) -> dict:
    """Parse a vis_pose predictions JSON → ``{image_stem: (kp, scores)}``.

    ``kp`` is the 308-keypoint ``[[x, y], ...]`` list of the first
    detected instance; ``scores`` the matching confidences."""
    data = json.loads(Path(json_path).read_text())
    out: dict = {}
    for fr in data.get("frames", []):
        inst = fr.get("instances", [])
        if inst:
            out[Path(fr["image_name"]).stem] = (
                inst[0]["keypoints"], inst[0]["keypoint_scores"])
    return out


def _segment_dist(xs: np.ndarray, ys: np.ndarray, a, b) -> np.ndarray:
    """Per-pixel distance from grid (xs, ys) to the segment a→b."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-6:
        return np.hypot(xs - ax, ys - ay)
    t = np.clip(((xs - ax) * dx + (ys - ay) * dy) / L2, 0.0, 1.0)
    return np.hypot(xs - (ax + t * dx), ys - (ay + t * dy))


def pose_arm_mask(seg: np.ndarray, kp, sc, *, min_score: float = 0.3,
                  radius_frac: float = 0.033) -> np.ndarray:
    """Boolean arm mask from pose keypoints — a tube around each arm.

    The 0.4b part-seg labels only the forearm/hand as an arm class; the
    *upper* arm reads as torso. With arms down that leak is harmless,
    but with arms spread it drags every chest slice out to the wrists.
    This cuts a fixed-radius tube along only the **shoulder→elbow**
    segment — the upper arm. The forearm and hand are already removed by
    the part-seg arm class; the elbow→wrist segment is *not* tubed
    because a bent arm brings the wrist back over the hips and a tube
    there would slice the torso. The radius is a small fraction of body
    height (≈ an upper-arm half-width).

    Intended for the **front** view only — in a side view the arm
    overlaps the torso in projection and a tube would cut real body.

    COCO body indices in the Goliath-308 set: shoulders 5/6, elbows 7/8.
    """
    h, w = seg.shape
    body = seg > 0
    rr = np.where(body.any(axis=1))[0]
    body_h = float(rr.max() - rr.min()) if len(rr) > 1 else float(h)
    radius = radius_frac * body_h
    ys, xs = np.mgrid[0:h, 0:w]
    mask = np.zeros((h, w), dtype=bool)
    for si, ei in ((5, 7), (6, 8)):
        s, e = kp[si], kp[ei]
        if min(sc[si], sc[ei]) < min_score:
            continue                       # weak arm — leave to the seg
        mask |= _segment_dist(xs, ys, s, e) < radius
    return mask


def pointmap_profile(
    ply: Path,
    seg_path: Path,
    photo: Path,
    height_cm: float,
    view: str,
    *,
    pose_kp=None,
    pose_sc=None,
    seg_backend: str = "rvm",
    band: int = 3,
    min_px: int = 40,
) -> SilhouetteProfile:
    """Metric width/depth profile of one view from its Sapiens pointmap.

    ``view`` is ``"front"`` (extent = body width) or ``"side"`` (extent =
    body depth). Both are the pointmap's X-extent — the image-horizontal
    axis — measured arm-free and rescaled so the body's vertical span
    equals ``height_cm``. The result is a
    :class:`~tailor_twin.fit.silhouette.SilhouetteProfile`, identical in
    shape to ``silhouette.extract_profile``'s output.
    """
    import cv2
    from .silhouette import _largest_component, load_silhouette

    seg = np.load(seg_path)
    h, w = seg.shape
    pm = _load_pointmap(ply, h, w)

    rvm, _ = load_silhouette(str(photo), backend=seg_backend)
    rvm = cv2.resize(rvm.astype(np.uint8), (w, h),
                     interpolation=cv2.INTER_NEAREST) > 0
    body = (seg > 0) & rvm                       # seg ∩ matte: drop stray edges
    if body.sum() < 500:
        raise SystemExit(f"{view}: empty body mask after seg ∩ matte")
    # Arm removal: the seg arm classes catch the forearm/hand; a pose
    # keypoint tube (when supplied) additionally cuts the upper arm the
    # seg mislabels as torso — essential when the arms are spread.
    arm_free = body & ~np.isin(seg, list(DOME29_ARM_CLASSES))
    if pose_kp is not None and pose_sc is not None:
        arm_free &= ~pose_arm_mask(seg, pose_kp, pose_sc)
    # Keep only the main torso blob — drops any stray mislabelled pixels
    # stranded out where the arms were.
    torso = _largest_component(arm_free)

    rows = np.where(body.any(axis=1))[0]
    r0, r1 = int(rows.min()), int(rows.max())

    # Metric scale: pointmap Y-extent of the body → true height.
    ymet = pm[..., 1][body]
    y_lo, y_hi = np.percentile(ymet, [0.5, 99.5])
    scale = height_cm / float(y_hi - y_lo)

    n = r1 - r0 + 1
    y_norm = np.zeros(n)
    extent = np.zeros(n)
    center = np.full(n, 0.5)
    valid = np.zeros(n, dtype=bool)
    for k, row in enumerate(range(r0, r1 + 1)):
        # y_norm: image row r0 = crown (1.0), r1 = feet (0.0).
        y_norm[k] = 1.0 - (row - r0) / max(r1 - r0, 1)
        lo = max(row - band, 0)
        hi = min(row + band + 1, h)
        m = torso[lo:hi]
        if m.sum() < min_px:
            continue
        xs = pm[lo:hi, :, 0][m]
        # scale already carries cm-per-metre (height_cm / body-Y-metres).
        extent[k] = (np.percentile(xs, 99) - np.percentile(xs, 1)) * scale
        valid[k] = True

    order = np.argsort(y_norm)                    # ascending feet→crown
    return SilhouetteProfile(
        y_norm=y_norm[order], extent_cm=extent[order],
        center_frac=center[order], valid=valid[order],
        height_px=float(r1 - r0), view=view)


def pointmap_arm_profile(
    ply: Path,
    seg_path: Path,
    height_cm: float,
    *,
    n_bins: int = 10,
) -> np.ndarray:
    """Metric arm width profile (cm, shoulder→wrist) from a pointmap.

    The arm sits at an angle, so a horizontal slice over-reads its
    girth. The arm's pointmap points are projected to the metric (x, y)
    plane and :func:`~tailor_twin.fit.silhouette._axis_profile` bins them
    along the limb's own principal axis — the same measurement
    ``silhouette_betas._arm_extents`` runs on the mesh arm, so target and
    model compare like-for-like.

    The larger-area arm (upper + lower, hand excluded) is measured; both
    arms share betas so one suffices.
    """
    from .silhouette import _axis_profile

    seg = np.load(seg_path)
    h, w = seg.shape
    pm = _load_pointmap(ply, h, w)

    left = np.isin(seg, list(DOME29_LEFT_ARM))
    right = np.isin(seg, list(DOME29_RIGHT_ARM))
    arm = left if left.sum() >= right.sum() else right
    if arm.sum() < n_bins * 4:
        return np.full(n_bins, np.nan)

    body = seg > 0
    ymet = pm[..., 1][body]
    y_lo, y_hi = np.percentile(ymet, [0.5, 99.5])
    scale = height_cm / float(y_hi - y_lo)

    ax_, ay_ = np.where(arm)
    xm = pm[arm][:, 0] * scale
    ym = pm[arm][:, 1] * scale
    # Orient: bin 0 must be the shoulder (top of the image, lowest row).
    # y_up = "metric y increases upward" — true when pm-y anti-correlates
    # with image row.
    y_up = bool(np.corrcoef(ym, ay_.astype(np.float64))[0, 1] < 0)
    pts = np.stack([xm, ym], axis=1)
    widths = _axis_profile(pts, n_bins, y_up=y_up)
    # End bins are unreliable — the shoulder bin catches the deltoid /
    # torso attachment, the last bins catch spread fingers. The SMPL-X
    # arm region ends do not line up with them either. Drop them so only
    # the mid-arm (bicep / forearm) enters the fit residual.
    widths[0] = np.nan
    widths[-2:] = np.nan
    return widths
