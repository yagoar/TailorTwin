"""CLI: run the Seamly catalog extractor on a fit.npz.

Runs every recipe in ``RECIPES`` + ``FORMULAS``, optionally re-poses
the bent-arm codes (L01/L02/L04 and the L03 formula), and writes any
combination of CSV / JSON / SMIS / OBJ artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import smplx

from .bent_arm import (
    DEFAULT_ELBOW_AXIS,
    DEFAULT_ELBOW_FLEX_DEG,
    DEFAULT_SHOULDER_FORWARD_DEG,
    repose_bent_arm,
)
from .exports import (
    PersonalInfo,
    write_csv,
    write_obj,
    write_smis_from_catalog,
)
from .landmarks import build_landmark_set
from .seamly_catalog import RECIPES
from .seamly_extractor import extract_catalog


BENT_ARM_CODES: tuple[str, ...] = ("L01", "L02", "L04")


# Mapping from landmark-editor line names → SMPL-X anchor landmarks used by
# the PlanarGirth recipes. Overriding the Y of each anchor pins the slice
# plane (and downstream landmarks like waist_cf/cb/side) to the photo Y.
_LM_TO_ANCHORS: dict[str, tuple[str, ...]] = {
    "bust":      ("bust_level",),
    "underbust": ("lowbust_level",),
    "waist":     ("waist_string", "waist_cf", "waist_cb",
                  "waist_side_left", "waist_side_right"),
    "highhip":   ("high_hip_level",),
    "hip":       ("hip_level",),
}


def _body_top_h(seg_path: Path) -> tuple[float, float]:
    """Top Y + height (in px) of the body in a Sapiens part-seg map."""
    seg = np.load(seg_path)
    if seg.ndim == 3:
        seg = (seg.argmax(0) if seg.shape[0] < seg.shape[-1]
               else seg.argmax(-1))
    ys = np.where(seg > 0)[0]
    return float(ys.min()), float(ys.max() - ys.min())


def _y_overrides_from_landmarks(
    landmarks_json: Path, front_seg: Path, side_seg: Path,
    smplx_verts: np.ndarray,
) -> dict[str, float]:
    """Convert photo-pixel landmark Y → world-frame mesh Y per anchor.

    Each landmark line in the JSON is a fraction of the photo body bbox
    (top-of-hair → toes). Map that same fraction onto the mesh's full
    Y span (feet → scalp) to get the slice height to feed downstream.
    """
    data = json.loads(landmarks_json.read_text())
    lines = data.get("lines_y") or {}
    f_top, f_h = _body_top_h(front_seg)
    s_top, s_h = _body_top_h(side_seg)
    y_lo = float(smplx_verts[:, 1].min())
    y_hi = float(smplx_verts[:, 1].max())
    span = y_hi - y_lo

    overrides: dict[str, float] = {}
    for lm_name, anchors in _LM_TO_ANCHORS.items():
        by_view = lines.get(lm_name) or {}
        if not by_view:
            continue
        # Side preferred (cleaner profile, no arm clutter on body Y).
        if by_view.get("side") is not None:
            frac = (by_view["side"] - s_top) / s_h
        elif by_view.get("front") is not None:
            frac = (by_view["front"] - f_top) / f_h
        else:
            continue
        mesh_y = y_lo + (1.0 - frac) * span
        for a in anchors:
            overrides[a] = mesh_y
    return overrides


def _print_table(values: dict, label: str = "seamly_code", unit: str = "cm") -> None:
    print(f"{label:<40} {'value (' + unit + ')':>10}")
    print("-" * 52)
    for k in sorted(values):
        print(f"{k:<40} {values[k]:>10.2f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Extract measurements from a fit.")
    p.add_argument("fit_npz", type=Path)
    p.add_argument("--model-folder", default="data/body_models")
    p.add_argument(
        "--gender", default=None,
        help="SMPL-X model gender. Defaults to the gender persisted in "
             "the fit npz; legacy fits without that field fall back to "
             "female.")
    p.add_argument("--num-betas", type=int, default=100)
    p.add_argument("--show-skipped", action="store_true")
    p.add_argument(
        "--save-seamly-json",
        type=Path,
        help="Write {seamly_code: value_cm} JSON",
    )
    p.add_argument(
        "--save-csv",
        type=Path,
        help="Write CSV: code, seamly_name, value_cm",
    )
    p.add_argument(
        "--save-obj",
        type=Path,
        help="Write fitted SMPL-X body mesh as Wavefront OBJ (for CLO3D)",
    )
    p.add_argument(
        "--save-smis",
        type=Path,
        help="Write SeamlyMe .smis directly (no intermediate JSON)",
    )
    p.add_argument(
        "--person-given-name", default=None,
        help="Sewer given name for the SMIS <personal> block. "
             "Defaults to the value persisted in the fit npz.",
    )
    p.add_argument(
        "--person-family-name", default=None,
        help="Sewer family name for the SMIS <personal> block. "
             "Defaults to the value persisted in the fit npz.",
    )
    p.add_argument(
        "--person-birth-date", default=None,
        help="ISO date yyyy-mm-dd for the SMIS <personal> block. "
             "Defaults to the value persisted in the fit npz.",
    )
    p.add_argument(
        "--person-gender", default=None,
        help="Sewer gender for the SMIS <personal> block "
             "(female / male / unknown). Defaults to the fit npz's "
             "gender field.",
    )
    p.add_argument(
        "--smis-template",
        type=Path,
        default=Path.home() / "seamly2d" / "templates" /
                "all_measurements_template.smis",
        help="Reference .smis whose measurement order is preserved",
    )
    p.add_argument(
        "--no-bent-arm",
        action="store_true",
        help="Skip the bent-arm re-pose pass (L01/L02/L03/L04 then "
             "fall through as A-pose values, which are incorrect).",
    )
    p.add_argument(
        "--landmarks", type=Path, default=None,
        help="JSON from scripts/landmark_editor.py — overrides the mesh-"
             "derived Y of bust_level / lowbust_level / waist_string / "
             "high_hip_level / hip_level so girths are sliced at the "
             "photo-Y you actually placed in the editor.",
    )
    p.add_argument(
        "--front-seg", type=Path, default=None,
        help="Sapiens front_seg.npy — required with --landmarks to map "
             "photo pixel Y → body-bbox fraction.",
    )
    p.add_argument(
        "--side-seg", type=Path, default=None,
        help="Sapiens side_seg.npy — required with --landmarks.",
    )
    p.add_argument(
        "--waist-y", type=float, default=None,
        help="World-frame Y (metres) of the detected waist-string elastic. "
             "Overrides the SMPL-X anatomical waist Y for every waist-"
             "anchored landmark (waist_cf, waist_cb, waist_side_left/right "
             "and everything that derives from them).",
    )
    p.add_argument(
        "--waist-y-from", type=Path, default=None,
        help="JSON file written by waist_string.detect_waist_y "
             "(`{ \"y_m\": float, ... }`). Reads y_m; equivalent to "
             "--waist-y but persists the detection metadata alongside.",
    )
    p.add_argument(
        "--waist-height-cm", type=float, default=None,
        help="Tape-measured waist height (floor → natural waist, vertical, "
             "cm). Resolved against THIS fit's mesh (waist Y = mesh min Y + "
             "height), so it stays valid after clean-fit / ring-deform "
             "re-centre the body — unlike an absolute --waist-y. "
             "Precedence: --waist-y > --waist-height-cm > --waist-y-from.",
    )
    p.add_argument(
        "--landmark-vid", action="append", default=None, metavar="NAME=VID",
        help="Override a base landmark's SMPL-X vertex id, repeatable, e.g. "
             "--landmark-vid acromion_left=4447. Corrects a mis-placed "
             "landmark without editing references/smplx_landmark_review.json. "
             "A *_left override auto-mirrors to *_right.",
    )
    p.add_argument(
        "--bent-elbow-flex-deg", type=float, default=DEFAULT_ELBOW_FLEX_DEG,
        help=f"Elbow flex angle for the bent-arm override "
             f"(default {DEFAULT_ELBOW_FLEX_DEG}°).",
    )
    p.add_argument(
        "--bent-elbow-axis", type=str, default=DEFAULT_ELBOW_AXIS,
        help="Elbow rotation axis in the L_Elbow local frame "
             f"(default {DEFAULT_ELBOW_AXIS!r} = forearm forward in the "
             "Seamly pose).",
    )
    p.add_argument(
        "--bent-shoulder-forward-deg", type=float,
        default=DEFAULT_SHOULDER_FORWARD_DEG,
        help="Extra L_Shoulder forward rotation (around world X) so the "
             "forearm doesn't collide with the torso when bent.",
    )
    args = p.parse_args(argv)

    # Photo-derived per-girth Y overrides (from manual landmark editor).
    y_overrides: dict[str, float] | None = None
    if args.landmarks is not None:
        if args.front_seg is None or args.side_seg is None:
            raise SystemExit(
                "--landmarks requires --front-seg AND --side-seg "
                "(to map photo pixel Y → body fraction).")
        y_overrides = _y_overrides_from_landmarks(
            args.landmarks, args.front_seg, args.side_seg,
            np.load(args.fit_npz)["smplx_vertices"])
        print(f"landmark Y overrides: "
              f"{ {k: round(v, 4) for k, v in y_overrides.items()} }")

    fit = np.load(args.fit_npz)
    verts = fit["smplx_vertices"].astype(np.float32)
    joints = (fit["smplx_joints"].astype(np.float32)
              if "smplx_joints" in fit.files else None)

    # Resolve waist-Y override (--waist-y > --waist-height-cm > JSON file).
    # --waist-height-cm is floor-relative and resolved against THIS mesh,
    # so it needs the fit's vertices — hence resolution happens here, after
    # the npz load.
    waist_y_override: float | None = args.waist_y
    if waist_y_override is None and args.waist_height_cm is not None:
        from .landmarks import waist_y_from_height
        waist_y_override = waist_y_from_height(verts, args.waist_height_cm)
        print(f"waist height {args.waist_height_cm:.1f} cm above floor "
              f"→ Y override {waist_y_override:.4f} m")
    if waist_y_override is None and args.waist_y_from is not None:
        from ..preprocess.waist_string import WaistStringDetection
        waist_y_override = WaistStringDetection.from_json(args.waist_y_from).y_m
    if waist_y_override is not None:
        print(f"waist Y override: {waist_y_override:.4f} m")
    from ..fit.fit import fit_gender, fit_person_info
    gender = args.gender or fit_gender(fit)

    npz_person = fit_person_info(fit)
    personal = PersonalInfo(
        given_name=(args.person_given_name
                    if args.person_given_name is not None
                    else npz_person["person_given_name"]),
        family_name=(args.person_family_name
                     if args.person_family_name is not None
                     else npz_person["person_family_name"]),
        birth_date=(args.person_birth_date
                    if args.person_birth_date is not None
                    else npz_person["person_birth_date"]),
        gender=(args.person_gender if args.person_gender else gender),
    )
    bm = smplx.create(
        model_path=args.model_folder, model_type="smplx",
        gender=gender, num_betas=args.num_betas,
        use_pca=False, batch_size=1,
    )
    faces = np.asarray(bm.faces, dtype=np.int32)

    # Landmark vertex-id overrides (hand-corrections). A *_left override
    # auto-mirrors to *_right via the template's left/right vertex map.
    vid_overrides: dict[str, int] | None = None
    if args.landmark_vid:
        from ..fit.clean_fit import build_symmetry_map
        sym = build_symmetry_map(
            bm.v_template.detach().cpu().numpy().astype(np.float64))
        vid_overrides = {}
        for spec in args.landmark_vid:
            if "=" not in spec:
                raise SystemExit(
                    f"--landmark-vid expects NAME=VID, got {spec!r}")
            name, vid = spec.split("=", 1)
            name, vid = name.strip(), int(vid)
            vid_overrides[name] = vid
            if name.endswith("_left"):
                vid_overrides[name[:-len("_left")] + "_right"] = int(sym[vid])
        print(f"landmark vid overrides: {vid_overrides}")

    landmarks_set = build_landmark_set(
        verts, joints=joints, faces=faces,
        waist_y_override=waist_y_override, y_overrides=y_overrides,
        vid_overrides=vid_overrides, gender=gender,
    )
    cat = extract_catalog(verts, faces, joints=joints,
                          waist_y_override=waist_y_override,
                          gender=gender, landmarks=landmarks_set)
    # Bent-arm override: L01/L02/L04 (and the L03 formula) need an
    # elbow-flexed mesh. Re-pose the SMPL-X body, recompute those
    # codes on the bent verts, then overwrite the A-pose values.
    if not args.no_bent_arm and "body_pose" in fit.files:
        try:
            pose = repose_bent_arm(
                fit, bm,
                elbow_flex_deg=args.bent_elbow_flex_deg,
                elbow_axis=args.bent_elbow_axis,
                shoulder_forward_deg=args.bent_shoulder_forward_deg,
            )
            bent_landmarks = build_landmark_set(
                pose.verts, joints=pose.joints, faces=faces,
                waist_y_override=waist_y_override,
                y_overrides=y_overrides, vid_overrides=vid_overrides,
                gender=gender,
            )
            for code in BENT_ARM_CODES:
                try:
                    cat.values[code] = float(
                        RECIPES[code].compute(
                            pose.verts, faces, bent_landmarks))
                except Exception as e:  # noqa: BLE001
                    print(f"bent {code}: {e}")
            if "L01" in cat.values and "L02" in cat.values:
                cat.values["L03"] = cat.values["L01"] - cat.values["L02"]
        except Exception as e:  # noqa: BLE001
            print(f"bent-arm override skipped: {e}")

    print("=" * 52)
    print("Seamly catalog extractor")
    print("=" * 52)
    _print_table(cat.values)
    print(f"\n{len(cat.values)} extracted   {len(cat.skipped)} skipped")
    if args.show_skipped:
        print("\nSkipped:")
        for k, reason in sorted(cat.skipped.items()):
            print(f"  {k}: {reason}")
    if args.save_seamly_json:
        args.save_seamly_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_seamly_json.write_text(
            json.dumps({k: float(v) for k, v in cat.values.items()}, indent=2)
        )
        print(f"\nsaved {args.save_seamly_json}")
    if args.save_csv:
        write_csv(cat.values, args.save_csv)
        print(f"saved {args.save_csv}")
    if args.save_smis:
        template = (args.smis_template
                    if args.smis_template and args.smis_template.is_file()
                    else None)
        write_smis_from_catalog(
            cat.values, args.save_smis, template, personal=personal)
        print(f"saved {args.save_smis}")

    if args.save_obj:
        write_obj(verts, faces, args.save_obj)
        print(f"saved {args.save_obj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
