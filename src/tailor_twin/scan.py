"""End-to-end TailorTwin pipeline: Stray capture → measurements.

Steps:
  1. Load Stray Scanner frames (rgb + depth + confidence + pose).
  2. Segment body per-frame (depth_threshold / rembg / rvm).
  3. Filter depth (confidence + range + bilateral).
  4. TSDF-fuse into a triangle mesh.
  5. Cleanup (largest component → hole fill → smooth → decimate).
  6. SMPL-X+D fit (parametric A-pose body).
  7. Extract Seamly catalog measurements (167+ codes).
  8. Re-pose for bent-arm L01/L02/L03/L04 and overwrite those codes.
  9. Write all artefacts: scan.obj, fit.npz, fit_body.obj, csv, smis,
     seamly_catalog.json, bent_arm.npz, bent_arm.json.

Run via the ``tailor-twin scan`` Typer command (see
``tailor_twin/cli.py``). For direct module invocation::

    python -m tailor_twin.scan data/captures/<name>/ \\
        --out-prefix data/results/<name>

Pass ``--skip-fusion`` to reuse an existing ``<prefix>_scan.obj`` —
useful when only re-running the fit/measure stages.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

from tailor_twin.io.stray_loader import load_capture
from tailor_twin.preprocess.depth_filter import (
    DEFAULT_MAX_DEPTH_MM,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_DEPTH_MM,
    apply_alpha_mask,
    filter_depth,
)
from tailor_twin.preprocess.segment import Segmenter, available_backends
from tailor_twin.reconstruct.cleanup import cleanup_mesh, rescale_to_stature
from tailor_twin.reconstruct.tsdf import (
    DEFAULT_SDF_TRUNC_M,
    DEFAULT_VOXEL_M,
    FusionInput,
    fuse_frames,
    fuse_frames_posegraph,
    save_mesh_obj,
)


def _iter_fusion_inputs(
    capture: Path,
    *,
    seg_backend: str,
    frame_stride: int,
    min_conf: int,
    min_depth_mm: int,
    max_depth_mm: int,
    bilateral: bool,
    alpha_threshold: float,
) -> Iterator[FusionInput]:
    """Stream segmented + filtered frames as FusionInput records."""
    segmenter = Segmenter(backend=seg_backend)
    needs_rgb = seg_backend != "depth_threshold"

    skipped_no_rgb = 0
    for i, frame in enumerate(
            load_capture(capture, decode_rgb=needs_rgb)):
        if i % frame_stride != 0:
            continue
        if frame.depth_mm is None:
            continue
        if needs_rgb and frame.rgb is None:
            # rgb.mp4 can decode fewer frames than odometry has rows (the
            # video stream ends a little early); those trailing frames have
            # no RGB, which the matting backends require. Skip them — they're
            # a handful at the tail, the fuse already has full coverage.
            skipped_no_rgb += 1
            continue
        filt = filter_depth(
            frame.depth_mm,
            confidence=frame.confidence,
            min_confidence=min_conf,
            min_depth_mm=min_depth_mm,
            max_depth_mm=max_depth_mm,
            bilateral=bilateral,
        )
        if (filt > 0).sum() < 200:
            continue  # frame mostly empty after filter — skip
        seg = segmenter.segment(frame.rgb, filt)
        masked = apply_alpha_mask(filt, seg.alpha_depth,
                                   threshold=alpha_threshold)
        if (masked > 0).sum() < 200:
            continue
        yield FusionInput(
            depth_mm=masked,
            intrinsics=frame.intrinsics,
            pose_c2w=frame.pose_cam_to_world,
        )
    if skipped_no_rgb:
        print(f"  ({skipped_no_rgb} trailing frame(s) skipped — no RGB "
              "decoded from rgb.mp4)")


def run(
    capture: Path,
    out_prefix: Path,
    *,
    seg_backend: str,
    voxel_m: float,
    sdf_trunc_m: float,
    frame_stride: int,
    min_conf: int,
    min_depth_mm: int,
    max_depth_mm: int,
    bilateral: bool,
    alpha_threshold: float,
    intrinsics_native_size: tuple[int, int] | None,
    skip_fusion: bool,
    pose_graph: bool,
    keyframe_stride: int,
    height_cm: float | None,
    tape_anchors: dict[str, float] | None,
    landmark_vids: list[str] | None,
    clean_fit: bool,
    apose_deg: float,
    model_folder: str,
    gender: str,
    num_betas: int,
    use_displacement: bool,
    smooth_d: bool,
    waist_height_cm: float | None = None,
    export_csv: bool = True,
    export_obj: bool = True,
    export_smis: bool = True,
    person_given_name: str = "",
    person_family_name: str = "",
    person_birth_date: str = "",
) -> int:
    """Run the full pipeline; return process exit code."""
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    scan_obj = out_prefix.with_name(out_prefix.name + "_scan.obj")
    fit_npz = out_prefix.with_name(out_prefix.name + "_smplx_fit.npz")
    fit_obj = out_prefix.with_name(out_prefix.name + "_fit_body.obj")
    csv_path = out_prefix.with_name(out_prefix.name + "_measurements.csv")
    json_path = out_prefix.with_name(out_prefix.name + "_seamly_catalog.json")
    smis_path = out_prefix.with_name(out_prefix.name + ".smis")

    # ---- 1. Stray → segmented/filtered frames → TSDF mesh.
    if not skip_fusion:
        # Stray reports fx/fy/cx/cy in the RGB camera's native resolution
        # (e.g. 1920x1440), but the depth map is 256x192. Open3D needs the
        # intrinsics in the depth pixel grid, so the native size must be
        # known to rescale. If the caller didn't pass it, read it straight
        # from rgb.mp4 — this is always the resolution Stray's intrinsics
        # are expressed in. Without this the projection is off by the
        # res ratio (~7.5x) and the fuse shatters into fragments.
        if intrinsics_native_size is None:
            import cv2
            _cap = cv2.VideoCapture(str(capture / "rgb.mp4"))
            _w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            _h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _cap.release()
            if _w > 0 and _h > 0:
                intrinsics_native_size = (_w, _h)
                print(f"  intrinsics native size auto-detected from rgb.mp4: "
                      f"{_w}x{_h} (depth is 256x192)")
        mode = "pose-graph" if pose_graph else "raw-odometry"
        print(f"[1/5] TSDF fusion (backend={seg_backend}, "
              f"voxel={voxel_m*1000:.1f}mm, {mode})")
        inputs = _iter_fusion_inputs(
            capture,
            seg_backend=seg_backend,
            frame_stride=frame_stride,
            min_conf=min_conf,
            min_depth_mm=min_depth_mm,
            max_depth_mm=max_depth_mm,
            bilateral=bilateral,
            alpha_threshold=alpha_threshold,
        )
        if pose_graph:
            mesh = fuse_frames_posegraph(
                inputs,
                voxel_length=voxel_m,
                sdf_trunc=sdf_trunc_m,
                intrinsics_native_size=intrinsics_native_size,
                keyframe_stride=keyframe_stride,
            )
        else:
            mesh = fuse_frames(
                inputs,
                voxel_length=voxel_m,
                sdf_trunc=sdf_trunc_m,
                intrinsics_native_size=intrinsics_native_size,
            )
        print("[2/5] cleanup")
        mesh = cleanup_mesh(mesh)
        if height_cm is not None:
            print(f"[2c] rescale to measured height ({height_cm:.1f} cm)")
            mesh, _factor = rescale_to_stature(mesh, height_cm / 100.0)
        save_mesh_obj(mesh, scan_obj)
        print(f"  wrote {scan_obj}")
    else:
        if not scan_obj.is_file():
            print(f"ERROR: --skip-fusion but {scan_obj} not found")
            return 1
        print(f"[1-2/5] reuse {scan_obj}")

    # ---- 3. SMPL-X fit.
    print(f"[3/5] SMPL-X fit (num_betas={num_betas})")
    import trimesh
    from tailor_twin.fit.fit import FitConfig, fit_scan, save_fit
    scan = trimesh.load(scan_obj, process=False)
    sv = np.asarray(scan.vertices, dtype=np.float32)
    sf = np.asarray(scan.faces, dtype=np.int32)

    cfg = FitConfig(
        model_folder=model_folder,
        gender=gender,
        num_betas=num_betas,
        device="cpu",
        use_displacement=use_displacement,
        use_smooth_displacement=smooth_d,
    )
    result = fit_scan(sv, cfg=cfg, verbose=True, scan_faces=sf)
    save_fit(result, fit_npz)
    print(f"  wrote {fit_npz}  (chamfer={result.final_chamfer:.6f})")

    # ---- 3a. Clean-fit: re-pose to the canonical A-pose and re-centre
    # (transl=0, global_orient=0) so the measured body is normalized, and —
    # when displacement is present — symmetrize it and drop head/hand scan
    # noise. Runs even without --use-displacement: the centering + A-pose
    # normalization matters regardless (a raw scan-pose fit sits off-centre,
    # which several surface-path measurements assume away). Measurement-safe.
    if clean_fit:
        extras = ("symmetrize + head/hand mask + " if use_displacement else "")
        print(f"[3a] clean-fit ({extras}A-pose {apose_deg:.0f}deg + recentre)")
        from tailor_twin.fit.clean_fit import clean_fit_npz
        clean_fit_npz(
            fit_npz, fit_npz, model_folder=model_folder, gender=gender,
            num_betas=num_betas, pose_deg=apose_deg)

    # ---- 3b. Tape-anchor girth calibration (optional).
    # A parametric/scan fit lands within a few cm on each girth. A handful
    # of tape-measured circumferences pin those rings exactly. Uniform
    # radial ring-scale corrects girth SIZE while preserving the scan's
    # real cross-section SHAPE (front/back/width ratio) — tape carries no
    # shape, only a scalar, so the proportions must come from the scan and
    # are kept here untouched. Runs before measurement so the downstream
    # CSV / SMIS / bent-arm all read the calibrated mesh.
    if tape_anchors:
        anchors = tape_anchors
        print(f"[3b] tape-anchor ring calibration ({len(anchors)} target(s))")
        tape_prefix = out_prefix.with_name(out_prefix.name + "_tape")
        cmd = [
            sys.executable, "-m", "tailor_twin.fit.ring_deform_cli",
            str(fit_npz),
            "--out-prefix", str(tape_prefix),
            "--num-betas", str(num_betas),
            "--model-folder", model_folder,
        ]
        for code, cm in anchors.items():
            cmd.extend(["--target", f"{code}={float(cm)}"])
        if waist_height_cm is not None:
            cmd.extend(["--waist-height-cm", f"{waist_height_cm:g}"])
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"  ring_deform_cli failed (exit {r.returncode})")
            return r.returncode
        # Redirect downstream measure + bent-arm at the calibrated fit.
        deformed_npz = tape_prefix.with_name(tape_prefix.name + "_smplx_fit.npz")
        if not deformed_npz.is_file():
            print(f"ERROR: expected calibrated fit not written: {deformed_npz}")
            return 1
        fit_npz = deformed_npz
        print(f"  calibrated fit: {fit_npz}")

    # ---- 4. Measurement extraction (incl. bent-arm override).
    print("[4/5] measurement extraction")
    cmd = [
        sys.executable, "-m", "tailor_twin.measure.cli", str(fit_npz),
        "--num-betas", str(num_betas),
        "--save-seamly-json", str(json_path),
    ]
    if export_csv:
        cmd.extend(["--save-csv", str(csv_path)])
    if export_smis:
        cmd.extend(["--save-smis", str(smis_path)])
    if export_obj:
        cmd.extend(["--save-obj", str(fit_obj)])
    if person_given_name:
        cmd.extend(["--person-given-name", person_given_name])
    if person_family_name:
        cmd.extend(["--person-family-name", person_family_name])
    if person_birth_date:
        cmd.extend(["--person-birth-date", person_birth_date])
    if gender:
        cmd.extend(["--person-gender", gender])
    if waist_height_cm is not None:
        cmd.extend(["--waist-height-cm", f"{waist_height_cm:g}"])
    for spec in (landmark_vids or []):
        cmd.extend(["--landmark-vid", spec])
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"  measure.cli failed (exit {r.returncode})")
        return r.returncode

    # ---- 4b. Measurement history: record the run and report drift vs the
    # same person's previous run. Repeatability is the metric that matters
    # for a personal tool — a code that jumped ≥ 1 cm between two scans of
    # the same body flags a capture/anchor problem NOW rather than as a
    # garment that doesn't fit. Best-effort: never fails the scan.
    try:
        from tailor_twin.history import (
            drift_rows, history_db_for, previous_values, record_run,
        )
        values = {k: float(v)
                  for k, v in json.loads(json_path.read_text()).items()}
        person = f"{person_given_name} {person_family_name}".strip() or "unknown"
        db = history_db_for(out_prefix)
        prev = previous_values(db, person=person)
        record_run(db, person=person, out_prefix=str(out_prefix),
                   values=values,
                   meta={"gender": gender, "num_betas": num_betas,
                         "tape_anchors": tape_anchors or {},
                         "waist_height_cm": waist_height_cm,
                         "height_cm": height_cm})
        if prev is None:
            print(f"  history: first recorded run for {person!r} ({db})")
        else:
            moved = drift_rows(values, prev, tol_cm=1.0)
            if moved:
                print(f"  history: {len(moved)} code(s) moved ≥ 1.0 cm vs "
                      f"{person!r}'s previous run:")
                for code, p_cm, c_cm, d in moved[:10]:
                    print(f"    {code}: {p_cm:.2f} → {c_cm:.2f} cm "
                          f"({d:+.2f})")
                if len(moved) > 10:
                    print(f"    … and {len(moved) - 10} more")
            else:
                print("  history: consistent with previous run (< 1.0 cm "
                      "drift on every shared code)")
    except Exception as e:  # noqa: BLE001 — history must never break a scan
        print(f"  history: skipped ({e})")

    # ---- 5. Bent-arm npz + json (for review viewer + audit).
    print("[5/5] bent-arm re-pose for L01/L02/L03/L04")
    cmd = [
        sys.executable, "-m", "tailor_twin.measure.extract_bent_arm",
        str(fit_npz),
        "--num-betas", str(num_betas), "--gender", gender,
        "--model-folder", model_folder,
    ]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"  extract_bent_arm.py failed (exit {r.returncode})")
        return r.returncode

    print("\nDONE.")
    print(f"  scan mesh:     {scan_obj}")
    print(f"  fit npz:       {fit_npz}")
    if export_obj:
        print(f"  fit body obj:  {fit_obj}")
    if export_csv:
        print(f"  csv:           {csv_path}")
    if export_smis:
        print(f"  smis:          {smis_path}")
    print(f"  catalog json:  {json_path}")
    print("\n3D viewer:  tailor-twin gui  (then pick the scan)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("capture", type=Path,
                   help="Stray Scanner capture folder")
    p.add_argument("--out-prefix", type=Path, required=True,
                   help="e.g. data/results/<name>")
    p.add_argument(
        "--seg-backend", default="rvm",
        choices=sorted({"rembg", "rvm", "depth_threshold"}),
        help=("body-segmentation backend (default rvm). "
              f"available now: {available_backends()}. "
              "rvm = torch.hub RVM mobilenetv3 — isolates the person from "
              "floor/walls (REQUIRED for a clean body; depth_threshold "
              "keeps everything in the depth window and the body fragments). "
              "rembg = `pip install rembg` (~180 MB, U2Net). "
              "depth_threshold = depth-only, no extra deps (debug only)."))
    p.add_argument("--voxel", type=float, default=DEFAULT_VOXEL_M)
    p.add_argument("--sdf-trunc", type=float, default=DEFAULT_SDF_TRUNC_M)
    p.add_argument("--frame-stride", type=int, default=1,
                   help="Integrate every Nth frame (1 = all).")
    p.add_argument("--min-confidence", type=int,
                   default=DEFAULT_MIN_CONFIDENCE,
                   help="Drop depth pixels below this Stray confidence tier.")
    p.add_argument("--min-depth-mm", type=int,
                   default=DEFAULT_MIN_DEPTH_MM)
    p.add_argument("--max-depth-mm", type=int,
                   default=DEFAULT_MAX_DEPTH_MM)
    p.add_argument("--no-bilateral", action="store_true",
                   help="Disable per-frame depth bilateral smooth.")
    p.add_argument("--alpha-threshold", type=float, default=0.5,
                   help="Min seg alpha to keep depth pixel.")
    p.add_argument("--intrinsics-native-w", type=int, default=None,
                   help="Width of the pixel grid in which Stray reports "
                        "fx/fy/cx/cy. Omit if intrinsics already match "
                        "the depth resolution.")
    p.add_argument("--intrinsics-native-h", type=int, default=None)
    p.add_argument("--skip-fusion", action="store_true",
                   help="Reuse an existing <prefix>_scan.obj.")
    p.add_argument(
        "--pose-graph", action="store_true",
        help="EXPERIMENTAL drift-corrected fusion: refine per-keyframe poses "
             "via Open3D multiway registration (ICP odometry + loop-closure "
             "edges → global optimization) before TSDF integration. Removes "
             "the doubled/ghosted surfaces that raw-odometry integration "
             "bakes in over a multi-loop capture. Not yet validated on a real "
             "capture; default off.")
    p.add_argument(
        "--keyframe-stride", type=int, default=3,
        help="With --pose-graph: integrate every Nth kept frame as a "
             "keyframe. Pairwise ICP is O(n²), so a few dozen keyframes is "
             "the sweet spot (default 3).")
    p.add_argument(
        "--height", type=float, default=None,
        help="Tape-measured standing height in CM. Uniformly rescales the "
             "fused mesh so its floor→crown extent matches this number, "
             "anchoring out odometry global-scale drift. Requires the capture "
             "to cover crown→feet.")
    p.add_argument(
        "--waist-height", type=float, default=None,
        help="Tape-measured waist height in CM: vertical distance from the "
             "floor to the natural waist (where the string is tied). Pins "
             "the waist line of every waist-anchored measurement to this "
             "height above the mesh floor; without it the SMPL-X "
             "anatomical waist is used.")
    p.add_argument(
        "--clean-fit", action=argparse.BooleanOptionalAction, default=True,
        help="Post-fit cleanup: re-pose to the canonical A-pose and re-centre "
             "the body so it's normalized for measurement, and — with "
             "--use-displacement — also symmetrize the displacement and zero "
             "it on the head + hands (removes warped-face / noisy-finger "
             "artifacts; measurement-safe). Default on.")
    p.add_argument(
        "--apose-deg", type=float, default=30.0,
        help="Shoulder angle (degrees from vertical) for the canonical "
             "A-pose the cleaned/measured body is posed in. Default 30.")
    p.add_argument(
        "--tape-anchors", type=Path, default=None,
        help="JSON {seamly_code: cm} of tape-measured girths to hit exactly, "
             "e.g. '{\"G04\": 88, \"G07\": 70, \"G09\": 99}'. Runs a uniform "
             "radial ring-scale per girth AFTER the fit, correcting girth "
             "size while preserving the scan's cross-section shape. "
             "Deformable codes: G03 highbust, G04 bust, G05 underbust, "
             "G07 waist, G08 highhip, G09 hip; legs M03 thigh, M05 knee, "
             "M07 calf, M09 ankle (each leg scaled independently).")
    p.add_argument(
        "--landmark-vid", action="append", default=None, metavar="NAME=VID",
        help="Override a base landmark's SMPL-X vertex id (forwarded to the "
             "measure step), repeatable, e.g. --landmark-vid "
             "acromion_left=4447. A *_left override auto-mirrors to *_right.")
    p.add_argument(
        "--tape-anchor", action="append", default=None, metavar="CODE=CM",
        help="Inline tape girth target, repeatable, e.g. --tape-anchor "
             "G04=87.5 --tape-anchor M07=34. Merged with --tape-anchors; "
             "same deformable codes. Lets the GUI pass anchors without a "
             "temp file.")
    p.add_argument("--model-folder", default="data/body_models")
    p.add_argument("--gender", default="female")
    p.add_argument("--num-betas", type=int, default=300)
    p.add_argument("--use-displacement", action="store_true")
    p.add_argument("--smooth-d", action="store_true")
    p.add_argument(
        "--export-csv", action=argparse.BooleanOptionalAction, default=True,
        help="Write the Seamly catalog CSV (+ filtered named CSV).")
    p.add_argument(
        "--export-obj", action=argparse.BooleanOptionalAction, default=True,
        help="Write the fitted SMPL-X body as a Wavefront OBJ.")
    p.add_argument(
        "--export-smis", action=argparse.BooleanOptionalAction, default=True,
        help="Write the SeamlyMe .smis file.")
    p.add_argument("--person-given-name", default="")
    p.add_argument("--person-family-name", default="")
    p.add_argument(
        "--person-birth-date", default="",
        help="ISO date yyyy-mm-dd written into the SMIS <personal> block.")
    args = p.parse_args(argv)

    native_size = None
    if args.intrinsics_native_w and args.intrinsics_native_h:
        native_size = (args.intrinsics_native_w, args.intrinsics_native_h)

    # Tape anchors: merge the JSON file (--tape-anchors) with inline
    # --tape-anchor CODE=cm flags into one {code: cm} dict.
    anchor_dict: dict[str, float] = {}
    if args.tape_anchors is not None:
        if not args.tape_anchors.is_file():
            raise SystemExit(
                f"--tape-anchors file not found: {args.tape_anchors}")
        loaded = json.loads(args.tape_anchors.read_text())
        if not isinstance(loaded, dict):
            raise SystemExit("--tape-anchors must be a {code: cm} JSON object")
        anchor_dict.update({str(k): float(v) for k, v in loaded.items()})
    for spec in (args.tape_anchor or []):
        if "=" not in spec:
            raise SystemExit(f"--tape-anchor expects CODE=cm, got {spec!r}")
        code, val = spec.split("=", 1)
        anchor_dict[code.strip()] = float(val)
    # --height also pins the FINAL body: the pre-fit scan rescale gets the
    # fit close, but the parametric fit drifts a cm or two, so add A01 to the
    # ring-deform targets to Y-scale the output mesh to the exact stature.
    # An explicit --tape-anchor A01 wins.
    if args.height is not None:
        anchor_dict.setdefault("A01", float(args.height))

    # Provenance: record the exact configuration + code version + outcome
    # next to the artifacts, whether the run succeeds or dies mid-stage.
    from tailor_twin.manifest import utc_now_iso, write_manifest
    started = utc_now_iso()
    rc = 1  # what the manifest reports if run() raises
    try:
        rc = _run_with_args(args, native_size, anchor_dict)
    finally:
        try:
            mpath = write_manifest(args.out_prefix, config=vars(args),
                                   rc=rc, started=started,
                                   finished=utc_now_iso())
            print(f"  manifest:      {mpath}")
        except Exception as e:  # noqa: BLE001 — provenance is best-effort
            print(f"  manifest: skipped ({e})")
    return rc


def _run_with_args(args, native_size, anchor_dict) -> int:
    return run(
        capture=args.capture,
        out_prefix=args.out_prefix,
        seg_backend=args.seg_backend,
        voxel_m=args.voxel,
        sdf_trunc_m=args.sdf_trunc,
        frame_stride=args.frame_stride,
        min_conf=args.min_confidence,
        min_depth_mm=args.min_depth_mm,
        max_depth_mm=args.max_depth_mm,
        bilateral=not args.no_bilateral,
        alpha_threshold=args.alpha_threshold,
        intrinsics_native_size=native_size,
        skip_fusion=args.skip_fusion,
        pose_graph=args.pose_graph,
        keyframe_stride=args.keyframe_stride,
        height_cm=args.height,
        tape_anchors=anchor_dict or None,
        landmark_vids=args.landmark_vid,
        clean_fit=args.clean_fit,
        apose_deg=args.apose_deg,
        model_folder=args.model_folder,
        gender=args.gender,
        num_betas=args.num_betas,
        use_displacement=args.use_displacement,
        smooth_d=args.smooth_d,
        waist_height_cm=args.waist_height,
        export_csv=args.export_csv,
        export_obj=args.export_obj,
        export_smis=args.export_smis,
        person_given_name=args.person_given_name,
        person_family_name=args.person_family_name,
        person_birth_date=args.person_birth_date,
    )


if __name__ == "__main__":
    raise SystemExit(main())
