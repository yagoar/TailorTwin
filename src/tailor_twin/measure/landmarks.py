"""Landmark resolver: symbolic name -> 3D point on the fitted SMPL-X mesh.

Sources:
  - references/smplx_landmark_review.json  (verified vertex IDs per
    GUARDRAILS section 3; the user signed off on each in Blender)
  - this module's COMPOUND_LANDMARKS dict (midpoints, axes derived from
    base landmarks — no new vertex IDs invented)

The schema in merged.yaml uses dotted names like `landmarks.bust_apex_left`
and `landmarks.bust_apex_midpoint`. Resolution is mechanical: split on the
dot, look up the leaf name in the verified-IDs map or in COMPOUND_LANDMARKS.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


# Unit-axis normalisation guard for compound / dynamic landmark resolution.
# Smaller than any plausible axis magnitude (axes here are differences of
# body landmarks ~10-50 cm). Same role as `EPS_AXIS_NORM` in primitives.py.
EPS_AXIS_NORM = 1e-9


DEFAULT_REVIEW_JSON = Path("references/smplx_landmark_review.json")


# SMPL-X joint indices, in the order returned by smplx body_model().joints.
# Source: smplx.body_models.SMPLX. Body joints 0..21, then jaw, eyes, hands.
SMPLX_JOINT_INDEX = {
    "pelvis": 0, "L_Hip": 1, "R_Hip": 2, "spine1": 3, "L_Knee": 4,
    "R_Knee": 5, "spine2": 6, "L_Ankle": 7, "R_Ankle": 8, "spine3": 9,
    "L_Foot": 10, "R_Foot": 11, "neck": 12, "L_Collar": 13, "R_Collar": 14,
    "head": 15, "L_Shoulder": 16, "R_Shoulder": 17, "L_Elbow": 18,
    "R_Elbow": 19, "L_Wrist": 20, "R_Wrist": 21,
}


# Landmarks whose verified vertex IDs sit on heavily arm-driven vertices.
# Populating an entry here overrides the vertex with the named SMPL-X joint
# position (pose-invariant rotation centre).
#
# WARNING: joint positions are INTERNAL (rotation centres ~5cm inboard from
# the skin). Overriding e.g. acromion shifts surface-tape measurements like
# B01 width inward by ~9cm. Use only for landmarks where geodesic-endpoint
# stability under pose matters more than surface accuracy.
#
# Empty by default. The `joint.L_Shoulder` syntax remains available for
# explicit opt-in inside recipes.
JOINT_OVERRIDES: dict[str, str] = {}


# Compound landmarks: name -> (operation, [base_names])
# Resolved by computing on the base landmarks' 3D coordinates.
COMPOUND_LANDMARKS: dict[str, tuple[str, list[str]]] = {
    # bust_apex_left/right: synthesized landmark interpolated between two
    # SMPL-X canonical vertices because the mesh has no single vertex at
    # the Y where Yaiza's actual apex sits.
    #   vid 5646 (Y=-0.040, X=+0.094) — too HIGH on the fitted body
    #   vid 3230 (Y=-0.057, X=+0.091) — too LOW on the fitted body
    # lerp t=0.5 takes the midpoint: Y=-0.048, X=+0.093. Override the
    # verified-vid lookup in smplx_landmark_review.json; the JSON keeps
    # 3230 / 5993 as audit-trail for the closer of the two real vids.
    # lerp_vids_surf: lerp the two canonical apex vids (sets X/Y → J01
    # width + bust-line height), then pin Z to the local breast surface
    # so the marker sits on the skin rather than on the chord through the
    # curved breast. Fixes the J-code lines converging off the surface.
    "bust_apex_left": ("lerp_vids_surf", ["5646", "3230", "0.5"]),
    "bust_apex_right": ("lerp_vids_surf", ["8340", "5993", "0.5"]),
    "bust_apex_midpoint": ("midpoint", ["bust_apex_left", "bust_apex_right"]),
    # armpit_apex: the TOP of the armpit hollow (deepest point a tape
    # reaches when pushed up into the armpit), ~3 cm above underarm_left.
    # Distinct from underarm_left, which sits at the highbust/armscye ring
    # level (G03) — that one must stay put (G03 girth is tape-validated).
    # Used by H03 (armpit→waist side) and A04 (armpit height), which are
    # measured from the hollow, not the ring.
    "armpit_apex_left": ("alias_vid", ["4032"]),
    "armpit_apex_right": ("alias_vid", ["6779"]),
    # Back midline at exact underarm Y — vid 5947 gives X/Z, underarm_left
    # gives Y. Used by G03 highbust so the back arc stays parallel to the
    # floor at true underarm height.
    "highbust_back_cf_at_underarm_y": ("snap_y_to", ["5947", "underarm_left"]),
    "armscye_back_midpoint": (
        "midpoint",
        ["armscye_back_left", "armscye_back_right"],
    ),
    "armscye_front_midpoint": (
        "midpoint",
        ["armscye_front_left", "armscye_front_right"],
    ),
    "shoulder_neck_midpoint": (
        "midpoint",
        ["shoulder_neck_left", "shoulder_neck_right"],
    ),
    "waist_side_midpoint": (
        "midpoint",
        ["waist_side_left", "waist_side_right"],
    ),
    # waist_string: a horizontal ring sitting at the natural waist. As an
    # anchor *point* we use the centre-front waist (Aldrich ties the string
    # there). The plane it defines uses the y-coordinate of this point.
    "waist_string": ("alias", ["waist_cf"]),

    # "Level" landmarks — symbolic horizontal planes referenced in merged.yaml
    # and the Seamly catalog. We represent each as a single 3D point whose
    # y-coordinate defines the horizontal slice; x/z carry no meaning when
    # used as a plane origin.
    "bust_level": ("alias", ["bust_apex_midpoint"]),
    "upper_bust_level": ("alias", ["armscye_front_midpoint"]),
    "armpit_level": ("alias", ["upper_bust_level"]),  # per extraction_audit.md
    "highbust_level": ("alias", ["armfold_front_left"]),  # armfold height
    "crotch_level": ("alias", ["crotch_midpoint"]),
    "mid_knee_level": ("midpoint", ["knee_back_left", "knee_back_right"]),
    "ankle_level": ("midpoint",
                    ["ankle_bone_lateral_left", "ankle_bone_lateral_right"]),
    # lowbust_level: detected underbust crease (per-body). The
    # lowbust_apex vid 3855 was too high on SMPL-X canonical; the
    # `underbust_crease_left` dynamic landmark finds the inframammary
    # fold directly from the fitted mesh.
    "lowbust_level": ("alias", ["underbust_crease_left"]),
    # Verified vid 5616 ("mid_neck_front") sits at ~20-30% up the neck
    # axis (throat hollow), not 50%. Use a true 50% lerp between the
    # SMPL-X neck and head joints so G01 lands on the actual mid-neck.
    "mid_neck_level": ("lerp_joint", ["neck", "head", "0.5"]),
    # neck_base_level: front_neck_point sits at the throat hollow. At +2.5cm
    # the slice grazed the trapezius (truth diff +27%); at +5cm it overshoot
    # into the upper neck (−10%). +4cm is the sweet spot for the base of
    # the actual neck cylinder where a tape is normally placed.
    "neck_base_level": ("offset_y", ["front_neck_point", "0.04"]),
    # ankle_high_level: see DYNAMIC_LANDMARKS — min-girth search between
    # ankle bone and knee. Previous fixed +0.03m offset failed on taller
    # figures (carmen 174cm showed A11 +38% / M09 ankle_circ −11% drift
    # vs height-scaled yaiza). Defining as a dynamic landmark below.
    # Synthetic plumb-line waypoints — X/Z from one landmark, Y from a
    # horizontal plane. Used by PolylineChord recipes (e.g. H01) so the
    # tape goes straight down from neck-front to the bust-level line,
    # then across to waist_cf, instead of contouring over the bust.
    "fnp_at_bust_y": ("snap_y_landmark",
                       ["front_neck_point", "bust_level"]),
    "sn_at_bust_y_left": ("snap_y_landmark",
                           ["shoulder_neck_left", "bust_level"]),
    "c7_at_bust_y": ("snap_y_landmark", ["c7", "bust_level"]),
    "c7_at_highbust_y": ("snap_y_landmark", ["c7", "armfold_front_left"]),
    "sn_at_highbust_y_left": ("snap_y_landmark",
                                ["shoulder_neck_left", "armfold_front_left"]),
    "fnp_at_highbust_y": ("snap_y_landmark",
                            ["front_neck_point", "armfold_front_left"]),
    "waist_side_left_at_hip_y": ("snap_y_landmark",
                                      ["waist_side_left", "hip_level"]),
    "waist_side_left_at_floor": ("snap_y_landmark",
                                   ["waist_side_left", "floor_anchor"]),
    "waist_side_left_at_highhip_y": ("snap_y_landmark",
                                       ["waist_side_left", "high_hip_level"]),
    # high_hip_level resolves to the slice centroid (body-interior),
    # so Geodesic's nearest-vertex pick can land on the back. Snap the
    # waist front centre's X/Z to the high-hip Y so H30's endpoint is
    # forced onto the front body surface.
    "waist_cf_at_highhip_y": ("snap_y_landmark",
                                ["waist_cf", "high_hip_level"]),
    "armfold_back_left_at_waist_y": ("snap_y_landmark",
                                       ["armfold_back_left", "waist_cb"]),
    "underarm_left_at_bust_y": ("snap_y_landmark",
                                  ["underarm_left", "bust_level"]),
    "underarm_right_at_bust_y": ("snap_y_landmark",
                                   ["underarm_right", "bust_level"]),
    # Synthetic interior arm points at the underarm Y — used as origins
    # for LimbGirth so the slice plane cuts the arm tube, not the torso.
    "l_arm_at_underarm_y": ("snap_y_landmark",
                              ["joint.L_Shoulder", "underarm_left"]),
    # acromion_left dropped to underarm Y — used as L16 endpoint so the
    # geodesic from acromion routes along the TOP of the upper arm down
    # to the L11 ring level (top of arm at armpit Y).
    "acromion_left_at_underarm_y": ("snap_y_landmark",
                                      ["acromion_left", "underarm_left"]),
    # Projection of acromion_left onto the L11 slice plane (perpendicular
    # to the upper-arm axis at l_arm_at_underarm_y). Used as L16 endpoint:
    # mirror of acromion across the arm axis at L11 height — a point on
    # the L11 ring plane lined up perpendicular to the arm.
    "acromion_left_proj_l11": ("project_perp", [
        "acromion_left",
        "l_arm_at_underarm_y",
        "joint.L_Shoulder",
        "joint.L_Elbow",
    ]),
    # bicep_max_left: mirror of bicep_max_right (vid 6022 → vid 3259,
    # found by minimising |verts[i] - mirrored(verts[6022])|). Used by
    # L13 to slice the LEFT upper arm at the bicep apex.
    "bicep_max_left": ("alias_vid", ["3259"]),
    # I03 / I08 height anchors: 2/3 between two Y references.
    # SN→apex line extended down to waist Y (H06 chord endpoint).
    "sn_to_apex_at_waist_y_left": ("extend_to_y", [
        "shoulder_neck_left", "bust_apex_left", "waist_string"]),
    "i03_y_anchor": ("lerp_y", ["front_neck_point",
                                 "armfold_front_left",
                                 "0.6667"]),
    "i08_y_anchor": ("lerp_y", ["c7", "armfold_back_left", "0.6667"]),
    # I03 / I08 endpoints: armscye / armfold X/Z at the 2/3 Y level.
    "armscye_front_left_at_i03_y": (
        "snap_y_landmark", ["armscye_front_left", "i03_y_anchor"]),
    "armscye_front_right_at_i03_y": (
        "snap_y_landmark", ["armscye_front_right", "i03_y_anchor"]),
    "armfold_back_left_at_i08_y": (
        "snap_y_landmark", ["armfold_back_left", "i08_y_anchor"]),
    "armfold_back_right_at_i08_y": (
        "snap_y_landmark", ["armfold_back_right", "i08_y_anchor"]),
    "bust_apex_left_at_lowbust_y": ("snap_y_landmark",
                                      ["bust_apex_left", "lowbust_level"]),
    "bust_apex_left_at_waist_y": ("snap_y_landmark",
                                    ["bust_apex_left", "waist_string"]),
    # high_hip_level lives in DYNAMIC_LANDMARKS — anatomical search for
    # the abdomen's max forward protrusion (Seamly A12 / G08).
    # hip_level: midpoint of SMPL-X L_Hip and R_Hip joint Y. The joints
    # are the femur-head rotation centres = greater trochanter region,
    # CAESAR-regressed across betas so they scale with body shape and
    # height automatically. On yaiza this lands G09 hip_circ at 92.95 cm
    # vs 92.13 cm truth tape (+0.9 %) — beats the dpm crotch+7 cm rule
    # (90.14 cm, −2.2 %) and the prior fixed waist−20 cm offset
    # (90.68 cm, −1.6 %). Midpoint is symmetry-tolerant (L_Hip ≠ R_Hip
    # by ~5 mm even at rest pose). X/Z of the joint are interior
    # (rotation centre, ~5 cm inside the body); downstream consumers
    # only use Y so the interior X/Z is irrelevant.
    "hip_level": ("lerp_joint", ["L_Hip", "R_Hip", "0.5"]),
}


# Reserved for landmarks computed per fit (e.g. searching a region of the
# deformed mesh for an extremum). Empty by default — tested with a
# `bust_apex_left/right -> max-Z search` override which made J01 worse
# (search produced more-medial points than the verified vids because the
# SMPL-X+D Laplacian smoothness flattens the apex peak relative to the
# truetoform tape position). Keeping the infrastructure for future use.
DYNAMIC_LANDMARKS: dict[str, dict] = {
    # Floor anchor: the body's lowest Y. Used to project landmarks down
    # to floor level for plumb-line measurements that terminate at the
    # ground (e.g. M02 outseam).
    "floor_anchor": {"search": "min_y"},
    # ankle_high_level: narrowest leg girth between ankle bone and knee.
    # Replaces the prior fixed +0.03m offset (which drifted on figures
    # with different leg proportions — A11 +38% / M09 -11% on carmen
    # vs height-scaled yaiza). Search window is the lower 35% of the
    # ankle→knee span starting 1.5cm above the malleolus (skips the
    # bone widest point, stops before the calf belly).
    "ankle_high_level": {
        "search": "min_girth_y",
        "y_lower": "ankle_bone_lateral_left",
        "y_upper": "knee_back_left",
        "regions": ("left_leg",),
        "y_lower_offset": 0.015,
        "y_upper_frac_of_span": 0.35,
        "samples": 30,
        "x_ref": "ankle_bone_lateral_left",
    },
    # high_hip_level: the highhip ring, taped 8 cm below the natural
    # waist. The search runs in a tight 1 cm band centred on
    # waist − 8 cm, so the slice is pinned to that height rather than
    # drifting to the abdomen apex (which sits lower and wider).
    "high_hip_level": {
        "search": "max_front_z_y",
        "y_lower": "waist_string",
        "y_upper": "waist_string",
        "y_lower_offset": -0.085,
        "y_upper_offset": 0.075,
        "regions": ("torso",),
        "x_midline_ref": "waist_cf",
        "x_midline_band": 0.05,
        "samples": 8,
    },
    # (hip_level moved back to COMPOUND_LANDMARKS — see crotch+0.07 entry.)
    # Waist front body point at the bust apex X — used as the endpoint
    # of J04 (bust apex to waist at apex X on the G07 line).
    "waist_front_at_apex_x_left": {
        "search": "body_at_xy",
        "x_ref": "bust_apex_left",
        "y_ref": "waist_string",
        "x_band": 0.02,
        "y_band": 0.01,
        "front_only": True,
    },
    # Body surface point on the G04 (bust) line at the SN_L X column.
    # Endpoint for H14 (neck side straight down to bust line).
    "bust_front_at_sn_x_left": {
        "search": "body_at_xy",
        "x_ref": "shoulder_neck_left",
        "y_ref": "bust_level",
        "x_band": 0.02,
        "y_band": 0.01,
        "front_only": True,
    },
    # Body surface point on the G03 (highbust = armfold_front Y) line at
    # the SN_L X column. Endpoint for H15.
    "highbust_front_at_sn_x_left": {
        "search": "body_at_xy",
        "x_ref": "shoulder_neck_left",
        "y_ref": "armfold_front_left",
        "x_band": 0.02,
        "y_band": 0.01,
        "front_only": True,
    },
    # Body surface point on the G07 (waist) line at SN_L's X column.
    # Used as the lower endpoint of H05 (neck side straight down to
    # waist front) so the line drops vertically from SN at SN_x.
    "waist_front_at_sn_x_left": {
        "search": "body_at_xy",
        "x_ref": "shoulder_neck_left",
        "y_ref": "waist_string",
        "x_band": 0.02,
        "y_band": 0.01,
        "front_only": True,
    },
    # Underbust crease (inframammary fold). Detected per-body by scanning
    # the anterior surface profile below the bust apex for the steepest
    # negative dZ/dY — the point where the breast tissue meets the
    # ribcage. Falls back gracefully if too few samples are found.
    # H16 endpoint: closest point on the G03 polyline to the 3D line
    # through SN_L and bust_apex_left. The endpoint sits on the actual
    # G03 curve where the SN→apex trajectory meets it.
    "h16_endpoint_left": {
        "search": "intersect_line_with_recipe",
        "line_a": "shoulder_neck_left",
        "line_b": "bust_apex_left",
        "recipe": "G03",
    },
    # H15 endpoint: where SN_L's vertical column meets the UPPER branch
    # of the G03 polyline on the front of the body. G03 dips low over
    # the bust front, so an unconstrained closest-point pick lands on
    # that dip; we filter to (x near SN_L.x) ∧ (z > 0) and then take
    # the highest-Y candidate — that's the actual SN-column G03
    # crossing, not the bust-front low point.
    "h15_endpoint_left": {
        "search": "intersect_line_with_recipe",
        "line_a": "shoulder_neck_left",
        "line_b": "sn_at_highbust_y_left",
        "recipe": "G03",
        "front_only": True,
        "x_band": 0.025,
        "prefer_y_max": True,
    },
    # H07 endpoint: same idea as H15 but on the body midline — vertical
    # column at the front_neck_point X dropped to its G03 crossing.
    # G03 (highbust arc) dips at the cleavage, so we again filter to
    # FNP's X column + front-only + take the upper-Y branch so the
    # plumb stops at the upper highbust line, not the bust-front dip.
    "h07_endpoint_front": {
        "search": "intersect_line_with_recipe",
        "line_a": "front_neck_point",
        "line_b": "fnp_at_highbust_y",
        "recipe": "G03",
        "front_only": True,
        "x_band": 0.025,
        "prefer_y_max": True,
    },
    # J03 endpoint: closest point on the G05 polyline to the vertical
    # line through bust_apex_left (apex straight down). Lands where
    # the inframammary G05 ring meets the apex's X/Z column.
    "j03_endpoint_left": {
        "search": "intersect_line_with_recipe",
        "line_a": "bust_apex_left",
        "line_b": "bust_apex_left_at_lowbust_y",
        "recipe": "G05",
    },
    # H06 lower endpoint: body surface at waist Y, at the X column
    # of the SN→apex line extended to waist Y. Lets the lower segment
    # of H06 "go in toward the body" instead of continuing in air at
    # the apex's forward Z.
    "h06_endpoint_left": {
        "search": "body_at_xy",
        "x_ref": "sn_to_apex_at_waist_y_left",
        "y_ref": "waist_string",
        "x_band": 0.025,
        "y_band": 0.012,
        "front_only": True,
    },
    # Body midline at bust Y plane — used as the centre point of G04
    # for P09's geodesic loop. Tight X band so the search lands on
    # the sternum/cleavage midline, not the inner edge of a breast.
    "bust_front_cf": {
        "search": "body_at_xy",
        "x_ref": "waist_cf",
        "y_ref": "bust_level",
        "x_band": 0.010,
        "y_band": 0.008,
        "front_only": True,
    },
    # J03 lower endpoint: actual body surface point at G05 (lowbust) Y
    # with X near the bust apex. Replaces the in-air snap_y_landmark
    # so the geodesic lands on the body, not floating below the bust.
    "bust_apex_left_at_lowbust_y_body": {
        "search": "body_at_xy",
        "x_ref": "bust_apex_left",
        "y_ref": "lowbust_level",
        "x_band": 0.025,
        "y_band": 0.012,
        "front_only": True,
    },
    "underbust_crease_left": {
        "search": "underbust_crease",
        "apex": "bust_apex_left",
        # Search 3-12 cm below apex (covers small to large busts).
        "min_offset": 0.03,
        "max_offset": 0.12,
        # Fold = where breast has receded `drop_fraction` of its full
        # depth (apex Z minus chest-wall Z, both measured automatically).
        # 0.7 lands at the actual inframammary crease on small/medium busts;
        # tune up for larger busts if needed.
        "drop_fraction": 0.5,
        "min_drop": 0.005,  # floor for flat chests
    },
}


# Per-leaf fallback resolution that only kicks in when the LandmarkSet's
# gender is not "female". The female-tuned vertex IDs in
# `references/smplx_landmark_review.json` land on the correct anatomy
# for the female SMPL-X model but slide up/down the male / neutral
# topology (the bust_apex vids end up near the male armpit, for
# instance). Each entry here points at a dynamic search whose result
# replaces the female anchor for the named landmark only on non-female
# fits.
MALE_FALLBACK_LANDMARKS: dict[str, dict] = {
    # Chest-fullest plane for male / neutral. The female bust_apex
    # midpoint sits ~4cm above the male armpit on the SMPL-X mesh and
    # produces an empty torso slice, so PlanarGirth at bust_level
    # fails outright. max_front_z_y on this band degenerated to belly
    # midline (abdomen fat protrudes more than the male chest centre).
    # max-girth on the torso slice does the right thing: peak girth
    # in [waist+8cm, armfold-1.5cm] lands at the nipple line on
    # Oscar (Y≈0.09 above waist) and matches tape truth within ~1%.
    "bust_level": {
        "search": "max_girth_y",
        "y_lower": "waist_string",
        "y_upper": "armfold_front_left",
        "y_lower_offset": 0.08,
        "y_upper_offset": 0.015,
        "regions": ("torso",),
        "x_ref": "waist_cf",
        "samples": 30,
    },
    # Seat plane for male / neutral. The female default (midpoint of
    # SMPL-X L_Hip / R_Hip joint Y) lands ~2.5cm below the seat curve
    # on male meshes. Max-girth in the same window degenerates to the
    # upper thigh on slim-hipped figures (Eric — narrow hips, full
    # thighs => peak girth at -0.351 instead of the seat curve).
    # Use the buttock-cap Y (max -Z = furthest backward protrusion)
    # which is the actual anatomical landmark a hip tape sits on,
    # regardless of hip / thigh ratio.
    # Oscar: G09 = 101.31 vs 101.27 truth   (+0.04 %).
    # Eric:  G09 = 95.82  vs 96.98 truth    (-1.19 %), and the ring
    #        now sits on the seat curve instead of the thigh.
    "hip_level": {
        "search": "max_back_z_y",
        "y_lower": "crotch_midpoint",
        "y_upper": "waist_string",
        "y_lower_offset": 0.02,
        "y_upper_offset": 0.04,
        "regions": ("torso", "left_leg", "right_leg"),
        "samples": 30,
    },
}


WAIST_VID_LANDMARKS: frozenset[str] = frozenset({
    "waist_cf",
    "waist_cb",
    "waist_side_left",
    "waist_side_right",
})
# Names whose Y must follow the detected waist-string Y when override is
# set. Compound landmarks that derive from these (waist_side_midpoint,
# waist_string, waist_string-anchored snap_y_landmark / extend_to_y /
# offset_y / body_at_xy entries) inherit the override automatically
# because they call back through `__getitem__`.


@dataclass(frozen=True)
class LandmarkSet:
    """Resolved 3D points for every named landmark on a specific mesh.

    `waist_y_override`: when set, overrides the Y coordinate of every
    landmark in `WAIST_VID_LANDMARKS`. The X/Z columns stay at the
    verified vertex IDs. Downstream landmarks (waist_string alias,
    waist-anchored snap_y_landmark / extend_to_y / offset_y / body_at_xy)
    inherit automatically.

    Use this to plumb the detected waist-string Y (from
    `tailor_twin.preprocess.waist_string.detect_waist_y`) into the
    measurement step so Aldrich #2, H05, J04, hip-level offsets, etc.
    are anchored at the user's tied waist instead of SMPL-X's CAESAR-
    learned anatomical waist.
    """

    verts: np.ndarray  # (V, 3) fitted SMPL-X vertices
    vertex_ids: dict[str, int]  # leaf name -> vertex id
    joints: np.ndarray | None = None  # (J, 3) fitted SMPL-X joint positions
    faces: np.ndarray | None = None  # (F, 3) triangle indices (needed by
    # dynamic landmarks that call into recipe polylines, e.g. the SN→apex
    # ↔ G03 intersection used by H16)
    waist_y_override: float | None = None  # world-frame Y from string detection
    y_overrides: dict[str, float] | None = None  # leaf -> world-frame Y.
    # Generic per-landmark Y override. Applied AFTER waist_y_override and
    # AFTER vertex/compound/dynamic resolution, so girth-slicing anchors
    # like bust_level / lowbust_level / waist_string / high_hip_level /
    # hip_level can be pinned to photo-derived heights from a manually
    # placed landmark JSON (scripts/landmark_editor.py).
    gender: str = "female"  # affects landmarks like bust_level where the
    # female-tuned vertex IDs land at the wrong anatomical level on
    # male / neutral meshes; non-female falls back to anatomical searches.
    override_leaves: frozenset[str] = frozenset()  # leaves whose vertex_id
    # came from an explicit --landmark-vid hand-pick. These win over
    # COMPOUND/DYNAMIC heuristics for the SAME leaf (e.g. pinning
    # bust_apex_left to a chosen surface vertex instead of the lerp).

    def __getitem__(self, name: str) -> np.ndarray:
        """Resolve `landmarks.<leaf>`, `joint.<NAME>`, or bare `<leaf>` to
        a 3D point."""
        if name.startswith("joint."):
            return self._joint(name.split(".", 1)[1])
        leaf = name.split(".", 1)[1] if name.startswith("landmarks.") else name
        pt = self._resolve(leaf)
        # Apply detected-waist-string Y override AT the end, after compound
        # / dynamic resolution. Y-only — preserves the verified X/Z of
        # waist_cf / waist_cb / waist_side_left / waist_side_right.
        if (self.waist_y_override is not None
                and leaf in WAIST_VID_LANDMARKS):
            pt = pt.copy()
            pt[1] = float(self.waist_y_override)
        if self.y_overrides and leaf in self.y_overrides:
            pt = pt.copy()
            pt[1] = float(self.y_overrides[leaf])
        return pt

    def _resolve(self, leaf: str) -> np.ndarray:
        """Internal resolver — same dispatch chain as `__getitem__` minus
        the waist-Y post-override (which is applied by the caller)."""
        # An explicit hand-picked vertex (--landmark-vid) wins over every
        # heuristic for that leaf — joint fallback, dynamic search, or
        # compound synthesis — so a corrected apex/acromion is honoured
        # even when the leaf is normally computed (e.g. a lerp).
        if leaf in self.override_leaves and leaf in self.vertex_ids:
            return self.verts[self.vertex_ids[leaf]]
        if leaf in JOINT_OVERRIDES and self.joints is not None:
            return self._joint(JOINT_OVERRIDES[leaf])

        if self.gender != "female" and leaf in MALE_FALLBACK_LANDMARKS:
            return self._dynamic(leaf, MALE_FALLBACK_LANDMARKS[leaf])

        if leaf in DYNAMIC_LANDMARKS:
            return self._dynamic(leaf, DYNAMIC_LANDMARKS[leaf])

        # COMPOUND_LANDMARKS overrides vertex_ids when the same leaf is in
        # both. Lets us synthesize a landmark position (e.g. lerp between
        # two vids) without removing the verified vid record from the JSON.
        if leaf in COMPOUND_LANDMARKS:
            op, bases = COMPOUND_LANDMARKS[leaf]
            handler = COMPOUND_OPS.get(op)
            if handler is None:
                raise ValueError(f"unknown compound op {op!r} for {leaf!r}")
            return handler(self, bases)

        if leaf in self.vertex_ids:
            return self.verts[self.vertex_ids[leaf]]

        raise KeyError(
            f"landmark {leaf!r} is neither a verified vertex ID nor a "
            "compound landmark. Add it to references/smplx_landmark_review.json "
            "(via Blender review) or to COMPOUND_LANDMARKS in this file."
        )

    def has(self, name: str) -> bool:
        if name.startswith("joint."):
            return (self.joints is not None
                    and name.split(".", 1)[1] in SMPLX_JOINT_INDEX)
        leaf = name.split(".", 1)[1] if name.startswith("landmarks.") else name
        if leaf in JOINT_OVERRIDES and self.joints is not None:
            return True
        return (leaf in self.vertex_ids or leaf in COMPOUND_LANDMARKS
                or leaf in DYNAMIC_LANDMARKS)

    def _dynamic(self, name: str, spec: dict) -> np.ndarray:
        """Resolve a landmark by searching the fitted mesh per the spec.
        See `DYNAMIC_SEARCHES` for the registered search types."""
        v = self.verts
        mask = _dynamic_prefilter(self, spec, v)
        search = spec["search"]
        handler = DYNAMIC_SEARCHES.get(search)
        if handler is None:
            raise ValueError(f"unknown search {search!r}")
        return handler(self, name, spec, mask)

    def _joint(self, joint_name: str) -> np.ndarray:
        if self.joints is None:
            raise KeyError(
                f"joint.{joint_name} requested but LandmarkSet was built "
                "without joints — pass `joints=fit['smplx_joints']` to "
                "build_landmark_set."
            )
        if joint_name not in SMPLX_JOINT_INDEX:
            raise KeyError(f"unknown SMPL-X joint name {joint_name!r}")
        return self.joints[SMPLX_JOINT_INDEX[joint_name]]


def load_vertex_ids(path: Path | str = DEFAULT_REVIEW_JSON) -> dict[str, int]:
    """Read the verified vertex IDs (with any status — confirmed, corrected,
    mirrored). Skipped entries are dropped."""
    raw = json.loads(Path(path).read_text())
    out: dict[str, int] = {}
    for name, rec in raw.items():
        if rec.get("status") == "skipped":
            continue
        out[name] = int(rec["vertex_id"])
    return out


def build_landmark_set(
    fitted_verts: np.ndarray,
    review_json: Path | str = DEFAULT_REVIEW_JSON,
    joints: np.ndarray | None = None,
    faces: np.ndarray | None = None,
    waist_y_override: float | None = None,
    y_overrides: dict[str, float] | None = None,
    vid_overrides: dict[str, int] | None = None,
    gender: str = "female",
) -> LandmarkSet:
    """Construct a LandmarkSet from a fitted SMPL-X mesh + verified IDs.

    `joints` is the (J, 3) array from a fit (e.g. fit['smplx_joints']);
    when supplied, it enables `joint.X` resolution and the JOINT_OVERRIDES
    fallback (acromion -> L_Shoulder, etc.).

    `waist_y_override` (world-frame Y, metres): replace the Y of every
    landmark in `WAIST_VID_LANDMARKS`. Typical source: the detected
    waist-string elastic Y from `tailor_twin.preprocess.waist_string`.

    `gender` selects between female-tuned vertex anchors (`bust_level` =
    `bust_apex_midpoint`) and anatomical searches for non-female bodies
    where those vids land off-target (`bust_level` = max front-Z slice
    on the chest, mirroring the high_hip search).

    `vid_overrides` ({leaf_name: vertex_id}) replaces the verified vertex ID
    of any base landmark — for hand-correcting a mis-placed landmark
    (e.g. acromion_left) without editing the review JSON. Overrides win
    over the JSON; downstream compound/dynamic landmarks that reference the
    overridden leaf follow automatically. An override on a leaf that is
    ITSELF compound/dynamic (e.g. bust_apex_left, normally a lerp) also
    wins — the hand-picked vertex replaces the synthesised position."""
    vertex_ids = load_vertex_ids(review_json)
    override_leaves: frozenset[str] = frozenset()
    if vid_overrides:
        vertex_ids = {**vertex_ids, **{k: int(v)
                                       for k, v in vid_overrides.items()}}
        override_leaves = frozenset(vid_overrides)
    return LandmarkSet(
        verts=fitted_verts,
        vertex_ids=vertex_ids,
        override_leaves=override_leaves,
        joints=joints,
        faces=faces,
        waist_y_override=waist_y_override,
        y_overrides=y_overrides,
        gender=gender,
    )


# ---------------------------------------------------------------------------
# Compound-op handlers. Each takes the LandmarkSet + the op's `bases` list
# from COMPOUND_LANDMARKS and returns the resolved 3D point.
# ---------------------------------------------------------------------------


def _op_offset_y(lm: LandmarkSet, bases: list[str]) -> np.ndarray:
    base = lm[bases[0]]
    dy = float(bases[1])
    return np.array([base[0], base[1] + dy, base[2]])


def _op_alias_vid(lm: LandmarkSet, bases: list[str]) -> np.ndarray:
    return lm.verts[int(bases[0])]


def _op_midpoint_of_vids(lm: LandmarkSet, bases: list[str]) -> np.ndarray:
    pts = np.stack([lm.verts[int(vid)] for vid in bases])
    return pts.mean(axis=0)


def _op_lerp_vids(lm: LandmarkSet, bases: list[str]) -> np.ndarray:
    vid_a, vid_b = int(bases[0]), int(bases[1])
    t = float(bases[2])
    return lm.verts[vid_a] * (1 - t) + lm.verts[vid_b] * t


def _op_lerp_vids_surf(lm: LandmarkSet, bases: list[str]) -> np.ndarray:
    """Lerp two vids, then lift the point onto the body surface.

    Keeps the lerp's X/Y — which set apex-to-apex width (J01) and the
    bust-line height (bust_level Y) — but replaces Z with the LOCAL front
    surface depth at that X/Y (the Z of the nearest front-hemisphere
    vertex in the X/Y plane). The raw lerp midpoint is a chord through
    the curved breast, so it can sit slightly off the skin; pinning Z to
    the local surface puts the marker — and every J-code line starting at
    it — on the body. Using the LOCAL nearest column (not the window max)
    avoids snapping Z to a more-projecting peak a couple cm away, which
    would make the marker poke out in front of its own column."""
    p = _op_lerp_vids(lm, bases).copy()
    v = lm.verts
    front = v[v[:, 2] > 0.0]  # front hemisphere only
    if len(front):
        d = np.hypot(front[:, 0] - p[0], front[:, 1] - p[1])
        p[2] = float(front[int(np.argmin(d)), 2])
    return p


def _op_snap_y_to(lm: LandmarkSet, bases: list[str]) -> np.ndarray:
    """Take X/Z from a vid, override Y with another landmark's Y."""
    pos = lm.verts[int(bases[0])].copy()
    pos[1] = lm[bases[1]][1]
    return pos


def _op_snap_y_landmark(lm: LandmarkSet, bases: list[str]) -> np.ndarray:
    """X/Z from one landmark, Y from another. Synthetic plumb-line waypoints."""
    a = lm[bases[0]]
    b = lm[bases[1]]
    return np.array([a[0], b[1], a[2]])


def _op_lerp_y(lm: LandmarkSet, bases: list[str]) -> np.ndarray:
    """3D lerp between two landmarks at fraction t."""
    a = lm[bases[0]]
    b = lm[bases[1]]
    t = float(bases[2])
    return a + t * (b - a)


def _op_extend_to_y(lm: LandmarkSet, bases: list[str]) -> np.ndarray:
    """Extend line a→b until its Y equals y_landmark.y."""
    a = lm[bases[0]]
    b = lm[bases[1]]
    y_target = lm[bases[2]][1]
    dy = b[1] - a[1]
    if abs(dy) < EPS_AXIS_NORM:
        return b.copy()
    t = (y_target - a[1]) / dy
    return a + t * (b - a)


def _op_lerp_joint(lm: LandmarkSet, bases: list[str]) -> np.ndarray:
    """3D lerp between two SMPL-X joints at fraction t."""
    a = lm._joint(bases[0])
    b = lm._joint(bases[1])
    t = float(bases[2])
    return a + t * (b - a)


def _op_project_perp(lm: LandmarkSet, bases: list[str]) -> np.ndarray:
    """Project `point` onto the plane through `origin` with normal axis_to -
    axis_from."""
    p = lm[bases[0]]
    o = lm[bases[1]]
    a = lm[bases[2]]
    b = lm[bases[3]]
    n = b - a
    nn = float(np.linalg.norm(n))
    if nn < EPS_AXIS_NORM:
        return p.copy()
    n = n / nn
    return p - ((p - o) @ n) * n


def _op_midpoint(lm: LandmarkSet, bases: list[str]) -> np.ndarray:
    pts = np.stack([lm[b] for b in bases])
    return pts.mean(axis=0)


def _op_alias(lm: LandmarkSet, bases: list[str]) -> np.ndarray:
    return lm[bases[0]]


COMPOUND_OPS: dict[str, Callable[[LandmarkSet, list[str]], np.ndarray]] = {
    "offset_y": _op_offset_y,
    "alias_vid": _op_alias_vid,
    "midpoint_of_vids": _op_midpoint_of_vids,
    "lerp_vids": _op_lerp_vids,
    "lerp_vids_surf": _op_lerp_vids_surf,
    "snap_y_to": _op_snap_y_to,
    "snap_y_landmark": _op_snap_y_landmark,
    "lerp_y": _op_lerp_y,
    "extend_to_y": _op_extend_to_y,
    "lerp_joint": _op_lerp_joint,
    "project_perp": _op_project_perp,
    "midpoint": _op_midpoint,
    "alias": _op_alias,
}


# ---------------------------------------------------------------------------
# Dynamic-search prefilter + handlers. The prefilter narrows the candidate
# vertex set by spec-level x_min/x_max/y_between bounds shared by most
# searches; each handler then picks from that masked set.
# ---------------------------------------------------------------------------


def _dynamic_prefilter(
    lm: LandmarkSet, spec: dict, v: np.ndarray,
) -> np.ndarray:
    mask = np.ones(len(v), dtype=bool)
    if "x_min" in spec:
        mask &= v[:, 0] > spec["x_min"]
    if "x_max" in spec:
        mask &= v[:, 0] < spec["x_max"]
    if "y_between" in spec:
        lower_name, upper_name = spec["y_between"]
        y_lo = float(lm[lower_name][1])
        y_hi = float(lm[upper_name][1])
        if y_lo > y_hi:
            y_lo, y_hi = y_hi, y_lo
        mask &= (v[:, 1] > y_lo) & (v[:, 1] < y_hi)
    if not mask.any():
        raise KeyError("dynamic landmark: no verts in search region")
    return mask


def _search_max_z(
    lm: LandmarkSet, name: str, spec: dict, mask: np.ndarray,
) -> np.ndarray:
    v = lm.verts
    idx = int(np.argmax(np.where(mask, v[:, 2], -np.inf)))
    return v[idx]


def _search_min_z(
    lm: LandmarkSet, name: str, spec: dict, mask: np.ndarray,
) -> np.ndarray:
    v = lm.verts
    idx = int(np.argmin(np.where(mask, v[:, 2], np.inf)))
    return v[idx]


def _search_min_y(
    lm: LandmarkSet, name: str, spec: dict, mask: np.ndarray,
) -> np.ndarray:
    v = lm.verts
    idx = int(np.argmin(np.where(mask, v[:, 1], np.inf)))
    return v[idx]


def _search_max_y(
    lm: LandmarkSet, name: str, spec: dict, mask: np.ndarray,
) -> np.ndarray:
    v = lm.verts
    idx = int(np.argmax(np.where(mask, v[:, 1], -np.inf)))
    return v[idx]


def _search_body_at_xy(
    lm: LandmarkSet, name: str, spec: dict, mask: np.ndarray,
) -> np.ndarray:
    """Body surface vertex closest to a target (X, Y).

    Defaults to picking the front-most vertex (max Z) within the X/Y
    band. Pass ``back_only: True`` to flip to the back-most (min Z)
    instead — needed for back-of-torso landmarks (h21/h23 endpoints).
    ``front_only`` and ``back_only`` are mutually exclusive; both
    unset is treated as ``front_only`` (legacy behaviour)."""
    v = lm.verts
    x = float(lm[spec["x_ref"]][0])
    y = float(lm[spec["y_ref"]][1])
    x_band = float(spec.get("x_band", 0.02))
    y_band = float(spec.get("y_band", 0.01))
    back_only = bool(spec.get("back_only"))
    # Legacy default: front-only. Explicit `back_only` flips the side.
    front_only = (not back_only) and spec.get("front_only", True)
    if front_only and back_only:
        raise KeyError(
            f"body_at_xy {name!r}: front_only and back_only are mutually exclusive")

    def _build_mask(scale: float) -> np.ndarray:
        mm = ((np.abs(v[:, 0] - x) < x_band * scale)
              & (np.abs(v[:, 1] - y) < y_band * scale))
        if front_only:
            mm &= v[:, 2] > 0
        elif back_only:
            mm &= v[:, 2] < 0
        return mm

    m = _build_mask(1.0)
    if not m.any():
        m = _build_mask(2.0)
    if not m.any():
        raise KeyError(f"body_at_xy {name!r}: no verts in band")
    if back_only:
        idx = int(np.argmin(np.where(m, v[:, 2], np.inf)))
    else:
        idx = int(np.argmax(np.where(m, v[:, 2], -np.inf)))
    return v[idx]


def _search_intersect_line_with_recipe(
    lm: LandmarkSet, name: str, spec: dict, mask: np.ndarray,
) -> np.ndarray:
    """Closest point on a recipe's polyline to the 3D line through two
    landmarks. Used by H16 (SN→apex × G03) and H15 (SN vertical × G03).

    Robustness goals (this is shared by every per-body intersection
    landmark, so an edge case in one figure must not skip a measurement
    on another):
      1. Scrub NaN / inf rows from the polyline before any geometry.
      2. Filters (`front_only`, `back_only`, `x_band`,
         `y_min`/`y_max`/`y_between`) restrict candidates BEFORE the
         perpendicular-distance pick. Conflicting filters (both
         `front_only` and `back_only`) raise immediately.
      3. Progressive widening: if the filter set is empty, retry with
         the band scaled 2x, then 4x. If still empty, drop `x_band`.
         If still empty, drop the side filter. Only then raise.
      4. Degenerate line (line_a ≈ line_b): fall back to nearest-point
         search around `line_a` instead of nearest-perpendicular.
      5. `prefer_y_max` / `prefer_y_min`: return the Y-extreme
         candidate after filters (used by H15 to grab the upper G03
         crossing rather than the bust-front dip).

    Spec keys (all optional except `recipe`, `line_a`, `line_b`):
      recipe         — Seamly code whose polyline supplies the curve.
      line_a, line_b — landmark names defining the 3D line.
      front_only     — keep curve points with Z > 0.
      back_only      — keep curve points with Z < 0.
      x_band         — abs(curve.x - line_a.x) < band (metres).
      y_min, y_max   — absolute Y bounds.
      y_between      — [low_landmark, high_landmark]; Y restricted to
                       that span (auto-swapped if reversed).
      prefer_y_max   — return highest-Y candidate after filters.
      prefer_y_min   — return lowest-Y candidate after filters.
    """
    # Local import — `seamly_catalog` imports primitives, which imports
    # this module. Lazy load breaks the cycle at call time.
    from .primitives import recipe_polyline
    from .seamly_catalog import RECIPES

    if spec.get("front_only") and spec.get("back_only"):
        raise ValueError(
            f"{name!r}: front_only and back_only are mutually exclusive")

    recipe_code = spec["recipe"]
    if recipe_code not in RECIPES:
        raise KeyError(f"{name!r}: unknown recipe {recipe_code!r}")
    recipe = RECIPES[recipe_code]
    curve = recipe_polyline(recipe, lm.verts, lm.faces, lm)
    if curve is None or len(curve) < 2:
        raise KeyError(
            f"{name!r}: recipe {recipe_code} has no polyline")

    # Scrub non-finite rows so a single solver glitch doesn't corrupt
    # argmin / argmax on the whole array.
    finite = np.isfinite(curve).all(axis=1)
    if finite.sum() < 2:
        raise KeyError(
            f"{name!r}: recipe {recipe_code} polyline has too few finite "
            "points")
    curve = curve[finite]

    a = lm[spec["line_a"]]
    b = lm[spec["line_b"]]

    use_front = bool(spec.get("front_only"))
    use_back = bool(spec.get("back_only"))
    has_x_band = "x_band" in spec
    x_band_base = float(spec["x_band"]) if has_x_band else 0.0

    y_low: float | None = None
    y_high: float | None = None
    if "y_min" in spec:
        y_low = float(spec["y_min"])
    if "y_max" in spec:
        y_high = float(spec["y_max"])
    if "y_between" in spec:
        lo_lm, hi_lm = spec["y_between"]
        y_lo_v = float(lm[lo_lm][1])
        y_hi_v = float(lm[hi_lm][1])
        if y_lo_v > y_hi_v:
            y_lo_v, y_hi_v = y_hi_v, y_lo_v
        y_low = y_lo_v if y_low is None else max(y_low, y_lo_v)
        y_high = y_hi_v if y_high is None else min(y_high, y_hi_v)

    def _filtered(band_scale: float, with_x: bool, with_side: bool,
                  with_y: bool) -> np.ndarray:
        k = np.ones(len(curve), dtype=bool)
        if with_side and use_front:
            k &= curve[:, 2] > 0
        if with_side and use_back:
            k &= curve[:, 2] < 0
        if with_x and has_x_band:
            k &= np.abs(curve[:, 0] - a[0]) < x_band_base * band_scale
        if with_y and y_low is not None:
            k &= curve[:, 1] >= y_low
        if with_y and y_high is not None:
            k &= curve[:, 1] <= y_high
        return k

    # Cascade: tighten → loosen. First with everything, then widen the
    # band, then drop the band, then drop the side, then drop the Y
    # window. Only fail when even the raw curve is empty (impossible —
    # we already checked len(curve) >= 2).
    for band_scale, with_x, with_side, with_y in (
        (1.0, True, True, True),
        (2.0, True, True, True),
        (4.0, True, True, True),
        (1.0, False, True, True),
        (1.0, False, False, True),
        (1.0, False, False, False),
    ):
        keep = _filtered(band_scale, with_x, with_side, with_y)
        if keep.any():
            break
    else:  # pragma: no cover — covered by the last fallback above
        raise KeyError(f"{name!r}: no recipe points survive any filter")
    sub = curve[keep]

    if spec.get("prefer_y_max"):
        return sub[int(np.argmax(sub[:, 1]))]
    if spec.get("prefer_y_min"):
        return sub[int(np.argmin(sub[:, 1]))]

    u = b - a
    L = float(np.linalg.norm(u))
    if L < EPS_AXIS_NORM:
        # Degenerate line — nearest-point fallback around line_a so the
        # caller still gets a sensible answer instead of `b.copy()`
        # (which may not lie on the recipe curve at all).
        d = np.linalg.norm(sub - a, axis=1)
        return sub[int(np.argmin(d))]
    u = u / L
    diff = sub - a
    proj_len = diff @ u
    proj = proj_len[:, None] * u
    perp = diff - proj
    d = np.linalg.norm(perp, axis=1)
    return sub[int(np.argmin(d))]


def _search_girth_y(
    lm: LandmarkSet, name: str, spec: dict, mask: np.ndarray,
    pick: str,
) -> np.ndarray:
    """Shared girth-extremum scan. ``pick`` is ``"min"`` or ``"max"``."""
    from .mesh_ops import (
        _build_loops, _convex_hull_perimeter, _loop_xz,
        _pick_loop_near_point, slice_mesh,
    )
    from .regions import region_vertex_mask

    if lm.faces is None:
        raise KeyError(f"{name!r}: faces required")
    y_lo = float(lm[spec["y_lower"]][1])
    y_hi = float(lm[spec["y_upper"]][1])
    span = y_hi - y_lo
    y_start = y_lo + float(spec.get("y_lower_offset", 0.0))
    if "y_upper_frac_of_span" in spec:
        y_end = y_lo + float(spec["y_upper_frac_of_span"]) * span
    else:
        y_end = y_hi - float(spec.get("y_upper_offset", 0.0))
    if y_start >= y_end:
        raise KeyError(f"{name!r}: empty Y range "
                       f"[{y_start:.3f}, {y_end:.3f}]")

    samples = int(spec.get("samples", 25))
    regions = tuple(spec.get("regions", ()))
    region_mask = region_vertex_mask(regions) if regions else None
    x_ref_pt = lm[spec["x_ref"]] if "x_ref" in spec else None
    y_axis = np.array([0.0, 1.0, 0.0])

    best_y: float | None = None
    best_centroid: np.ndarray | None = None
    if pick == "min":
        best_girth = np.inf
        improved = lambda g, b: g < b  # noqa: E731
    else:
        best_girth = -np.inf
        improved = lambda g, b: g > b  # noqa: E731

    for y in np.linspace(y_start, y_end, samples):
        origin = np.array([0.0, float(y), 0.0])
        segs = slice_mesh(lm.verts, lm.faces, origin, y_axis,
                          vertex_mask=region_mask)
        loops = _build_loops(segs)
        if not loops:
            continue
        loop = (_pick_loop_near_point(loops, x_ref_pt)
                if x_ref_pt is not None else loops[0])
        if loop is None or len(loop) < 3:
            continue
        g = float(_convex_hull_perimeter(_loop_xz(loop, y_axis)))
        if improved(g, best_girth):
            best_girth = g
            best_y = float(y)
            best_centroid = loop.mean(axis=0)
    if best_y is None:
        raise KeyError(f"{name!r}: no slices found in window")
    if x_ref_pt is not None:
        return np.array([x_ref_pt[0], best_y, x_ref_pt[2]])
    assert best_centroid is not None
    return np.array([best_centroid[0], best_y, best_centroid[2]])


def _search_min_girth_y(
    lm: LandmarkSet, name: str, spec: dict, mask: np.ndarray,
) -> np.ndarray:
    """Y of the narrowest convex-hull slice in a vertical window.

    Scans `samples` horizontal slices between `y_lower` and `y_upper`
    (optionally offset / fractionally clipped), restricted to vertices
    in `regions` (e.g. ("left_leg",) so the slice tube is just one leg).
    Picks the Y whose slice has the smallest convex-hull perimeter.

    Returns a point at that Y, with X/Z taken from `x_ref` if given,
    else the loop centroid. Used by `ankle_high_level` to find the
    "high ankle" — narrowest point above the malleolus, distinct from
    the ankle bone itself.

    Spec keys (only `y_lower` / `y_upper` required):
      y_lower, y_upper     — landmarks defining the Y window.
      regions              — region names (see regions.REGIONS).
      y_lower_offset       — metres to add to y_lower (clears the malleolus
                             bulge for ankle_high; default 0).
      y_upper_offset       — metres to subtract from y_upper (default 0).
      y_upper_frac_of_span — alternative to y_upper_offset: clip the
                             upper Y to y_lower + frac * (y_upper - y_lower).
                             Lets us scan the lower N% of the limb only.
      samples              — number of Y slices to evaluate (default 25).
      x_ref                — landmark whose X/Z is copied into the result
                             (and used to disambiguate when multiple loops
                             survive — e.g. both legs in a torso slice).
    """
    return _search_girth_y(lm, name, spec, mask, pick="min")


def _search_max_girth_y(
    lm: LandmarkSet, name: str, spec: dict, mask: np.ndarray,
) -> np.ndarray:
    """Y of the widest convex-hull slice in a vertical window.

    Mirror of `_search_min_girth_y`. Used for `bust_level` on male /
    neutral fits — picks the chest-fullest Y where horizontal girth
    peaks. On Oscar this lands at the nipple line (Y ~0.09 above
    waist), giving G04 ≈ 106 cm vs 106.25 cm tape (-0.3 %). The
    max-front-Z fallback we tried first picked the belly midline
    because abdomen fat sticks out more than the male chest centre.
    """
    return _search_girth_y(lm, name, spec, mask, pick="max")


def _scan_slices(lm, spec):
    """Iterate Y-plane slice loops for hip-style anatomical searches.

    Yields (y, loop) for each Y between `y_lower` and `y_upper`
    landmarks (with optional offsets / fractional clip), restricted to
    `regions`. Picks the torso loop when `torso` is among the regions
    so the buttock + abdomen surface comes through in one loop.
    """
    from .mesh_ops import (_build_loops, _pick_largest_loop,
                            _pick_torso_loop, slice_mesh)
    from .regions import region_vertex_mask

    if lm.faces is None:
        raise KeyError("hip search: faces required")
    y_lo = float(lm[spec["y_lower"]][1])
    y_hi = float(lm[spec["y_upper"]][1])
    y_start = y_lo + float(spec.get("y_lower_offset", 0.0))
    y_end = y_hi - float(spec.get("y_upper_offset", 0.0))
    if y_start >= y_end:
        raise KeyError(f"hip search: empty Y range "
                       f"[{y_start:.3f}, {y_end:.3f}]")
    samples = int(spec.get("samples", 25))
    regions = tuple(spec.get("regions", ("torso",)))
    region_mask = region_vertex_mask(regions) if regions else None
    y_axis = np.array([0.0, 1.0, 0.0])
    pick_torso = "torso" in regions
    for y in np.linspace(y_start, y_end, samples):
        origin = np.array([0.0, float(y), 0.0])
        segs = slice_mesh(lm.verts, lm.faces, origin, y_axis,
                          vertex_mask=region_mask)
        loops = _build_loops(segs)
        if not loops:
            continue
        loop = _pick_torso_loop(loops) if pick_torso else _pick_largest_loop(loops)
        if loop is None or len(loop) < 3:
            continue
        yield float(y), loop


def _search_extreme_z_y(
    lm: LandmarkSet, name: str, spec: dict, mask: np.ndarray,
    sign: int,
) -> np.ndarray:
    """Shared front/back-Z scan. ``sign=+1`` picks max Z (forward
    protrusion), ``sign=-1`` picks min Z (backward protrusion)."""
    x_band = spec.get("x_midline_band")
    x_mid = (float(lm[spec["x_midline_ref"]][0])
             if "x_midline_ref" in spec else 0.0)
    best_y: float | None = None
    best_score = -np.inf
    best_loop: np.ndarray | None = None
    for y, loop in _scan_slices(lm, spec):
        pts = loop
        if x_band is not None:
            keep = np.abs(pts[:, 0] - x_mid) < float(x_band)
            if not keep.any():
                continue
            pts = pts[keep]
        score = float((pts[:, 2] * sign).max())  # sign flips min→max
        if score > best_score:
            best_score = score
            best_y = y
            best_loop = loop
    if best_y is None or best_loop is None:
        raise KeyError(f"{name!r}: no slices found")
    centroid = best_loop.mean(axis=0)
    return np.array([centroid[0], best_y, centroid[2]])


def _search_max_back_z_y(
    lm: LandmarkSet, name: str, spec: dict, mask: np.ndarray,
) -> np.ndarray:
    """Y where the slice's back (min Z) protrusion is greatest. Mirror
    of `_search_max_front_z_y`, used for `hip_level` on male / neutral
    fits to anchor on the buttock cap regardless of thigh / hip ratio
    (max-girth lands on the thigh for slim-hipped figures)."""
    return _search_extreme_z_y(lm, name, spec, mask, sign=-1)


def _search_max_front_z_y(
    lm: LandmarkSet, name: str, spec: dict, mask: np.ndarray,
) -> np.ndarray:
    """Y where the slice's front (max Z) protrusion is greatest.

    Used for `high_hip_level` per the Seamly definition: "Highhip
    level, where front abdomen is most prominent" (G08 / A12). Optional
    `x_midline_band` (with `x_midline_ref` defaulting to centre-X = 0)
    restricts the max-Z search to slice points near the body midline
    so a wide hip flare can't outvote the belly.
    """
    return _search_extreme_z_y(lm, name, spec, mask, sign=+1)


def _search_max_lateral_x_y(
    lm: LandmarkSet, name: str, spec: dict, mask: np.ndarray,
) -> np.ndarray:
    """Y where the slice's lateral (|X|) extent is largest.

    Used for `hip_level` per the Seamly definition: "Hip where Hip
    protrusion is greatest" (G09, old_name `hips_excluding_protruding_abdomen`).
    Includes the leg regions in the slice mask so the lateral cap at
    the trochanter / seat is included (same convention as G09's
    PlanarGirth regions tuple).
    """
    best_y: float | None = None
    best_w = -np.inf
    best_loop: np.ndarray | None = None
    for y, loop in _scan_slices(lm, spec):
        w = float(loop[:, 0].max() - loop[:, 0].min())
        if w > best_w:
            best_w = w
            best_y = y
            best_loop = loop
    if best_y is None or best_loop is None:
        raise KeyError(f"max_lateral_x_y {name!r}: no slices found")
    centroid = best_loop.mean(axis=0)
    return np.array([centroid[0], best_y, centroid[2]])


def _search_underbust_crease(
    lm: LandmarkSet, name: str, spec: dict, mask: np.ndarray,
) -> np.ndarray:
    """Detect the inframammary fold per-body.

    1. Bust depth = (apex Z) - (chest-wall reference Z) sampled in a
       narrow X slab a few cm above the breast attachment.
    2. Scan Y from min_offset below apex down to max_offset, sampling
       max-Z of body verts at apex X.
    3. Crease = first Y where Z has dropped by drop_fraction of the
       bust depth, or the steepest-gradient Y as fallback.
    """
    v = lm.verts
    ref = lm[spec["apex"]]
    apex_y = float(ref[1])
    apex_x = float(ref[0])
    apex_z = float(ref[2])
    slab_dx_chest = float(spec.get("chest_slab_dx", 0.03))
    chest_y = apex_y + float(spec.get("chest_y_offset", 0.08))
    chest_y_band = float(spec.get("chest_y_band", 0.01))
    chest_mask = ((np.abs(v[:, 0] - apex_x) < slab_dx_chest)
                  & (np.abs(v[:, 1] - chest_y) < chest_y_band)
                  & (v[:, 2] > 0))
    if chest_mask.any():
        chest_z = float(v[chest_mask, 2].mean())
    else:
        chest_z = apex_z  # degenerate
    bust_depth = max(apex_z - chest_z, 0.0)
    drop_fraction = float(spec.get("drop_fraction", 0.5))
    min_drop = float(spec.get("min_drop", 0.005))
    fold_drop = max(bust_depth * drop_fraction, min_drop)

    y_top = apex_y - float(spec.get("min_offset", 0.02))
    y_bot = apex_y - float(spec.get("max_offset", 0.12))
    slab_dx = float(spec.get("slab_dx", 0.04))
    y_band = float(spec.get("y_band", 0.005))
    ys = np.linspace(y_top, y_bot, 60)
    zs: list[float] = []
    ys_kept: list[float] = []
    for y in ys:
        m = ((np.abs(v[:, 0] - apex_x) < slab_dx)
             & (np.abs(v[:, 1] - y) < y_band)
             & (v[:, 2] > 0))
        if not m.any():
            continue
        ys_kept.append(float(y))
        zs.append(float(v[m, 2].max()))
    if len(ys_kept) < 5:
        raise KeyError(f"underbust_crease {name!r}: too few samples")
    ys_arr = np.array(ys_kept)
    zs_arr = np.array(zs)
    z_top = zs_arr[0]
    below = np.where(z_top - zs_arr >= fold_drop)[0]
    if len(below):
        i = int(below[0])
    else:
        dz = np.gradient(zs_arr, ys_arr)
        i = int(np.argmin(dz))
    return np.array([apex_x, ys_arr[i], zs_arr[i]])


DYNAMIC_SEARCHES: dict[str, Callable[
    [LandmarkSet, str, dict, np.ndarray], np.ndarray]] = {
    "max_z": _search_max_z,
    "min_z": _search_min_z,
    "min_y": _search_min_y,
    "max_y": _search_max_y,
    "body_at_xy": _search_body_at_xy,
    "intersect_line_with_recipe": _search_intersect_line_with_recipe,
    "min_girth_y": _search_min_girth_y,
    "max_girth_y": _search_max_girth_y,
    "max_front_z_y": _search_max_front_z_y,
    "max_back_z_y": _search_max_back_z_y,
    "max_lateral_x_y": _search_max_lateral_x_y,
    "underbust_crease": _search_underbust_crease,
}
