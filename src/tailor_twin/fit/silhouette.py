"""Silhouette body profiling — the 3DLook Mobile Tailor geometry step.

3DLook's insight: two orthogonal photos (front + side), shot with the
phone held vertical (gyroscope-gated → near-orthographic projection),
plus the user's height as an absolute scale reference, fully constrain a
body's *outline*:

  * the **front** silhouette gives body **width** at every height Y;
  * the **side** silhouette gives body **depth** (front-to-back) at
    every height Y.

Unlike an image-feature regressor (SHAPY, SAM 3D Body) — which maps
pixels to a learned population-mean shape and so smooths a real body
toward the training prior — a silhouette IS the subject's actual
outline. No prior, no regression: width(Y) and depth(Y) are measured,
not guessed.

This module turns one segmented photo into a per-normalized-Y extent
profile. ``silhouette_fit_cli`` then scales an SMPL-X torso so each
horizontal slice matches the measured (width, depth) pair.

Normalization
-------------
Each silhouette is scaled by its own feet→head pixel span to the
user-supplied ``height_cm``. Front and side photos therefore need not be
shot at the same distance or zoom — both are mapped to the same absolute
cm axis, with Y = 0 at the feet and Y = 1 at the crown.

Arm handling
------------
The arms corrupt both views — in the front photo they fuse to the torso
above the armpit; in the side photo they hang in front of the belly and
inflate the measured depth. The clean fix is a body-*part* segmentation:
Sapiens part-seg labels arm + hand pixels with their own class ids, so
they can be deleted, leaving a torso+legs+head silhouette.

``load_silhouette`` therefore prefers a Sapiens ``*_seg.npy`` (arms
dropped → ``arm_free=True``). Given a plain photo it falls back to
``rvm``/``rembg`` matting (whole person, ``arm_free=False``); for that
case ``extract_profile`` keeps the front-view "central run with an arm
gap" heuristic, but the side view stays arm-contaminated — pass seg maps
for a trustworthy depth profile.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SilhouetteProfile:
    """Per-Y outline extent of one segmented photo.

    ``y_norm`` runs 0.0 (feet) → 1.0 (crown). ``extent_cm`` is the body
    extent on the photo's horizontal axis at that height — width for a
    front photo, depth for a side photo. ``center_frac`` is the extent
    run's mid-point as a fraction of the silhouette bounding-box width
    (0 = left edge, 1 = right edge); used to track left/right or
    front/back asymmetry.
    """
    y_norm: np.ndarray       # (N,) ascending, 0=feet 1=head
    extent_cm: np.ndarray    # (N,) body extent in cm
    center_frac: np.ndarray  # (N,) run centre, fraction of bbox width
    valid: np.ndarray        # (N,) bool — extent trustworthy at this Y
    height_px: float         # feet→head span in pixels
    view: str                # "front" | "side"


# Sapiens part-seg class ids for the two arms (incl. hands). Verified
# against this project's me_9626 capture: the two lateral mid-height
# blobs. Dropping them leaves a torso+legs+head silhouette.
SAPIENS_ARM_CLASSES = (6, 15)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 8-connected blob — discards stray seg
    speckle and any arm fragment left disconnected after removal."""
    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    if n < 2:
        return mask
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best


def load_silhouette(
    path: str, *,
    arm_classes: tuple[int, ...] = SAPIENS_ARM_CLASSES,
    backend: str = "rvm",
) -> tuple[np.ndarray, bool]:
    """Load a boolean body silhouette. Returns ``(mask, arm_free)``.

    ``path`` ending ``.npy`` → a Sapiens part-seg map: the silhouette is
    every non-background pixel minus ``arm_classes`` (arms + hands), so
    ``arm_free=True``. Any other path → an RGB photo segmented with the
    ``rvm``/``rembg`` matting backend (whole person, ``arm_free=False``).
    """
    if str(path).endswith(".npy"):
        seg = np.load(path)
        if seg.ndim != 2:
            raise SystemExit(
                f"seg map {path!r} must be a 2-D class map, got {seg.shape}")
        mask = seg > 0
        for c in arm_classes:
            mask &= seg != c
        return _largest_component(mask), True

    import cv2

    from ..preprocess.segment import Segmenter

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"could not read image {path!r}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    seg = Segmenter(backend=backend)
    # depth_mm is unused by rvm/rembg but the API requires a shape; pass
    # a zero array at the RGB resolution so the alpha is not resized.
    h, w = rgb.shape[:2]
    res = seg.segment(rgb, np.zeros((h, w), dtype=np.float32))
    alpha = res.alpha_rgb if res.alpha_rgb is not None else res.alpha_depth
    return alpha >= 0.5, False


def _runs(cols: np.ndarray) -> list[tuple[int, int]]:
    """Split a sorted array of foreground column indices into contiguous
    [start, end] runs (a gap of >1 px starts a new run)."""
    if len(cols) == 0:
        return []
    splits = np.where(np.diff(cols) > 1)[0]
    bounds = np.split(cols, splits + 1)
    return [(int(b[0]), int(b[-1])) for b in bounds]


def _row_extent(
    row_mask: np.ndarray, body_cx: float, *, pick: str,
) -> tuple[float, float, int] | None:
    """Extent of one scan-line. Returns ``(span_px, center_px, n_runs)``
    or None.

    ``pick`` chooses which foreground run the extent is taken from:

    * ``"central"`` — the run straddling ``body_cx`` (front torso): the
      arms, separate flanking runs in an A-pose, are excluded.
    * ``"widest"`` — the widest run (a single leg below the crotch,
      where the scan-line splits into two leg blobs).
    * ``"full"`` — leftmost-to-rightmost span (side torso, one blob).

    ``n_runs`` reports the foreground run count: for the front torso 3+
    means a clean arm | torso | arm split, 1 means fused arms.
    """
    cols = np.where(row_mask)[0]
    if len(cols) < 2:
        return None
    runs = _runs(cols)
    if pick == "full" or len(runs) == 1:
        lo, hi = cols[0], cols[-1]
    elif pick == "widest":
        lo, hi = max(runs, key=lambda r: r[1] - r[0])
    else:  # "central" — run containing body_cx, else the nearest
        inside = [r for r in runs if r[0] <= body_cx <= r[1]]
        if inside:
            lo, hi = max(inside, key=lambda r: r[1] - r[0])
        else:
            lo, hi = min(runs,
                         key=lambda r: abs((r[0] + r[1]) * 0.5 - body_cx))
    return float(hi - lo), float((hi + lo) * 0.5), len(runs)


def _median_smooth(a: np.ndarray, k: int = 5) -> np.ndarray:
    """Odd-window median filter — kills per-row segmentation jitter
    without rounding off real body curvature."""
    if k < 3:
        return a
    pad = k // 2
    padded = np.pad(a, pad, mode="edge")
    out = np.empty_like(a)
    for i in range(len(a)):
        out[i] = np.median(padded[i:i + k])
    return out


def extract_profile(
    mask: np.ndarray,
    height_cm: float,
    *,
    view: str,
    arm_free: bool = False,
    pick: str | None = None,
    n_slices: int = 256,
    smooth_k: int = 5,
) -> SilhouetteProfile:
    """Turn a boolean silhouette into a :class:`SilhouetteProfile`.

    The mask's foreground bounding box defines feet (bottom row) and
    crown (top row); that pixel span maps to ``height_cm``. ``view`` is
    ``"front"`` or ``"side"``.

    ``pick`` selects the run the extent is read from (see
    :func:`_row_extent`). When ``None`` it defaults per view —
    ``"central"`` for the front torso, ``"full"`` for the side torso.
    Leg slices below the crotch pass ``pick="widest"`` to read a single
    leg blob.

    ``arm_free`` — the mask already has the arms removed (from a part-seg
    map). Every row is then trustworthy. Otherwise a ``"central"`` front
    profile trusts only rows that split into ≥3 runs (a real arm gap).
    """
    if view not in ("front", "side"):
        raise ValueError(f"view must be 'front' or 'side', got {view!r}")
    if pick is None:
        pick = "central" if view == "front" else "full"
    ys, xs = np.where(mask)
    if len(ys) < 100:
        raise SystemExit(
            f"{view} silhouette nearly empty ({len(ys)} px) — "
            "segmentation failed; check the photo / backend")
    top, bot = int(ys.min()), int(ys.max())
    left, right = int(xs.min()), int(xs.max())
    height_px = float(bot - top)
    bbox_w = float(right - left) or 1.0
    body_cx = float(xs.mean())
    px_to_cm = height_cm / height_px

    y_norm = np.linspace(0.0, 1.0, n_slices)
    extent_px = np.zeros(n_slices)
    center_px = np.full(n_slices, body_cx)
    n_runs = np.zeros(n_slices, dtype=int)
    for i, g in enumerate(y_norm):
        # g = 0 at feet (bottom row), 1 at crown (top row).
        row = int(round(bot - g * height_px))
        row = min(max(row, 0), mask.shape[0] - 1)
        res = _row_extent(mask[row], body_cx, pick=pick)
        if res is not None:
            extent_px[i], center_px[i], n_runs[i] = res

    extent_px = _median_smooth(extent_px, smooth_k)
    # Trust rules: a "central" front profile on a raw photo trusts only
    # rows that split into ≥3 runs (arm | torso | arm — a real gap);
    # fused-arm rows would report torso+arm as the width. An arm-free
    # mask, or a "widest"/"full" profile, is trustworthy on every
    # non-empty row.
    if pick == "central" and not arm_free:
        valid = n_runs >= 3
    elif pick == "widest" and view == "front":
        # Front leg width: only trustworthy where the scan-line shows ≥2
        # runs (the two legs apart). One run = thighs touching → the run
        # spans both legs, not one.
        valid = n_runs >= 2
    else:
        valid = extent_px > 0
    return SilhouetteProfile(
        y_norm=y_norm,
        extent_cm=extent_px * px_to_cm,
        center_frac=(center_px - left) / bbox_w,
        valid=valid,
        height_px=height_px,
        view=view,
    )


def _axis_profile(pts: np.ndarray, n_bins: int, *, y_up: bool) -> np.ndarray:
    """Width of a limb sampled along its own principal axis.

    A limb (arm) sits at an arbitrary angle, so a horizontal slice would
    cut it obliquely and over-read its girth. PCA of the limb's points
    gives the major axis (shoulder→wrist) and the minor axis (width).
    Points are binned along the major axis; each bin's width is the
    minor-axis spread (5th–95th percentile, robust to stray pixels).

    ``pts`` is (N, 2) ``[x, y]``. ``y_up`` True for a 3-D mesh
    projection (Y increases upward), False for image pixels (row index
    increases downward) — it only fixes the shoulder→wrist orientation
    so bin 0 is always the shoulder. Returns ``n_bins`` widths."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < n_bins * 4:
        return np.full(n_bins, np.nan)
    c = pts.mean(axis=0)
    p = pts - c
    cov = (p.T @ p) / len(p)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    minor = evecs[:, int(np.argmin(evals))]
    t = p @ major
    s = p @ minor
    # Orient the major axis so bin 0 is the shoulder (the "up" end).
    up = pts[:, 1] if y_up else -pts[:, 1]
    if np.corrcoef(t, up)[0, 1] > 0:
        t = -t
    edges = np.linspace(t.min(), t.max(), n_bins + 1)
    widths = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (t >= edges[i]) & (t < edges[i + 1])
        if m.sum() >= 4:
            si = s[m]
            widths[i] = np.percentile(si, 95) - np.percentile(si, 5)
    return widths


def arm_profile_from_seg(
    seg: np.ndarray | str,
    height_cm: float,
    *,
    arm_class: int,
    n_bins: int = 10,
) -> np.ndarray:
    """Arm width profile (cm, shoulder→wrist) from a Sapiens part-seg.

    ``arm_class`` is the part-seg id of the arm to measure (6 / 15 for
    the two arms in this project's captures). The width is in the front
    image plane — for the roughly-circular arm cross-section that is a
    good proxy for girth (girth ≈ π·width). Scale comes from the full
    body's pixel height mapped to ``height_cm``."""
    if isinstance(seg, str):
        seg = np.load(seg)
    body = seg > 0
    ys = np.where(body.any(axis=1))[0]
    if len(ys) < 2:
        raise SystemExit("seg map has no body")
    px_to_cm = height_cm / float(ys.max() - ys.min())
    arm = _largest_component(seg == arm_class)   # drop stray seg pixels
    ay, ax = np.where(arm)
    if len(ay) < n_bins * 4:
        return np.full(n_bins, np.nan)
    pts = np.stack([ax, ay], axis=1)
    widths = _axis_profile(pts, n_bins, y_up=False) * px_to_cm
    # The arm class's shoulder cut and hand inclusion do not line up
    # with the SMPL-X left_arm region's ends — only the mid-arm bins
    # (bicep/forearm) compare like-for-like. Trim the boundary bins.
    widths[0] = np.nan
    widths[-2:] = np.nan
    return widths


def sample_extent_cm(profile: SilhouetteProfile, y_norm: float) -> float:
    """Linearly interpolate the body extent (cm) at a normalized height."""
    return float(np.interp(y_norm, profile.y_norm, profile.extent_cm))


def sample_valid(profile: SilhouetteProfile, y_norm: float) -> bool:
    """Is the extent at ``y_norm`` trustworthy? False inside any band the
    nearest sampled slices flagged invalid (e.g. fused-arm torso rows)."""
    i = int(np.argmin(np.abs(profile.y_norm - y_norm)))
    return bool(profile.valid[i])
