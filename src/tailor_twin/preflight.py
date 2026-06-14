"""Quick pre-flight sanity check for a Stray Scanner capture.

Run BEFORE the multi-minute TSDF + fit pipeline to catch broken
captures (low confidence, dropped frames, IMU divergence).

Usage:
    python scripts/inspect_stray.py data/captures/<name>/

Outputs (stdout):
    - frame count + duration
    - depth confidence-tier histogram (% per Stray tier 0/1/2)
    - depth valid-pixel ratio at the recommended filter window
    - per-frame depth median (catches "subject too far" captures)
    - camera pose track length + start/end translation delta
    - sample frame indices written to /tmp/<name>_inspect_<idx>.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tailor_twin.io.stray_loader import load_capture, validate_capture
from tailor_twin.preprocess.depth_filter import (
    DEFAULT_MAX_DEPTH_MM,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_DEPTH_MM,
    confidence_histogram,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("capture", type=Path)
    p.add_argument("--sample-frames", type=int, default=4,
                   help="Number of RGB frames to dump as PNG previews.")
    args = p.parse_args(argv)

    validate_capture(args.capture)

    conf_total: dict[int, int] = {0: 0, 1: 0, 2: 0}
    valid_pixel_ratios: list[float] = []
    depth_medians_mm: list[float] = []
    poses: list[np.ndarray] = []
    timestamps: list[float] = []
    sample_indices: list[int] = []

    frames = list(load_capture(args.capture, decode_rgb=True))
    if not frames:
        print("ERROR: no frames in capture")
        return 1

    every = max(1, len(frames) // max(args.sample_frames, 1))
    out_dir = Path("/tmp")

    for i, f in enumerate(frames):
        if f.confidence is not None:
            for k, v in confidence_histogram(f.confidence).items():
                conf_total[int(k)] = conf_total.get(int(k), 0) + int(v)
        d = f.depth_mm
        if d is not None:
            mask = (
                (d >= DEFAULT_MIN_DEPTH_MM)
                & (d <= DEFAULT_MAX_DEPTH_MM)
                & (f.confidence >= DEFAULT_MIN_CONFIDENCE
                   if f.confidence is not None else True)
            )
            valid_pixel_ratios.append(float(mask.mean()))
            kept = d[mask]
            if kept.size:
                depth_medians_mm.append(float(np.median(kept)))
        poses.append(f.pose_cam_to_world)
        timestamps.append(f.timestamp_s)
        if i % every == 0 and f.rgb is not None:
            import cv2
            out = out_dir / f"{args.capture.name}_inspect_{i:04d}.png"
            cv2.imwrite(str(out), f.rgb[..., ::-1])  # RGB → BGR for cv2
            sample_indices.append(i)

    n = len(frames)
    duration = (timestamps[-1] - timestamps[0]) if n > 1 else 0.0
    pose_translations = np.stack([T[:3, 3] for T in poses])
    drift = np.linalg.norm(pose_translations[-1] - pose_translations[0])
    track_len = float(np.linalg.norm(
        np.diff(pose_translations, axis=0), axis=1).sum())

    print(f"=== {args.capture.name} ===")
    print(f"frames:            {n}")
    print(f"duration:          {duration:.2f} s")
    print(f"fps (approx):      {n / max(duration, 1e-6):.1f}")
    total_conf = sum(conf_total.values()) or 1
    print(
        f"confidence pct:    "
        f"0={conf_total[0] / total_conf * 100:.1f}% "
        f"1={conf_total[1] / total_conf * 100:.1f}% "
        f"2={conf_total[2] / total_conf * 100:.1f}%")
    if valid_pixel_ratios:
        print(
            f"valid pixels:      "
            f"mean={np.mean(valid_pixel_ratios) * 100:.1f}% "
            f"min={np.min(valid_pixel_ratios) * 100:.1f}% "
            f"max={np.max(valid_pixel_ratios) * 100:.1f}% "
            f"(window: {DEFAULT_MIN_DEPTH_MM}-{DEFAULT_MAX_DEPTH_MM} mm, "
            f"conf ≥ {DEFAULT_MIN_CONFIDENCE})")
    if depth_medians_mm:
        print(
            f"subject distance:  "
            f"median={np.median(depth_medians_mm):.0f} mm "
            f"min={np.min(depth_medians_mm):.0f} mm "
            f"max={np.max(depth_medians_mm):.0f} mm")
    print(f"pose track length: {track_len:.2f} m")
    gap_pct = (drift / track_len * 100.0) if track_len > 1e-6 else 0.0
    print(f"start↔end gap:     {drift:.3f} m "
          f"({gap_pct:.0f}% of track length)")
    if sample_indices:
        print("sample frames:     " + ", ".join(
            f"/tmp/{args.capture.name}_inspect_{i:04d}.png"
            for i in sample_indices))

    # Health verdict. Only conditions that genuinely predict a bad fuse
    # fail the check. The start↔end gap is NOT one: it's large whenever the
    # orbit doesn't return to its start (a normal open sweep — a small
    # fraction of the track length), and is a true drift signal only when
    # the path was meant to close. Real odometry drift shows up as a
    # doubled/ghosted mesh, so it's a post-fuse visual check, not a
    # pre-fuse alarm — surfaced as a note below, never a failure.
    alarms = []
    if conf_total.get(2, 0) / total_conf < 0.20:
        alarms.append("< 20 % conf-2 depth — LiDAR noisy, recapture")
    if valid_pixel_ratios and np.mean(valid_pixel_ratios) < 0.05:
        alarms.append("< 5 % valid pixels — subject out of window")
    if alarms:
        print("\nALARMS:")
        for a in alarms:
            print(f"  ! {a}")
        return 2
    if drift > 0.50:
        print(f"\nnote: start↔end gap is {drift:.2f} m — normal for an open "
              "orbit that doesn't return to its start. Only a problem if the "
              "fused mesh comes out doubled/ghosted; check it in the viewer.")
    print("\nverdict: OK to proceed to TSDF fusion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
