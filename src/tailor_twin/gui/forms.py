"""Pure form helpers: slug, split name, validate, build subprocess args.

These functions only consume plain dicts (the route handler passes
``request.form.to_dict()``) so they're directly unit-testable without
spinning up Flask.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

from .config import (
    ENABLED_GENDERS,
    PIPELINE_PY,
    RUN_SCAN_ARGS,
    TAPE_GIRTHS,
    VALID_GENDERS,
)


def _parse_pos_float(raw: str | None) -> float | None:
    """Parse a positive float from a form field; None if blank/invalid."""
    s = (raw or "").strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if v > 0 else None


def nest_out_prefix(out_prefix: str) -> str:
    """Fold a run's artifacts into a per-run folder.

    The GUI field holds ``<results>/<stem>``; one run writes many files
    (``<stem>_smplx_fit.npz``, ``<stem>_tape_*`` …). Returning
    ``<results>/<stem>/<stem>`` drops them all into a ``<stem>/`` folder
    while keeping the prefixed, self-describing filenames. Idempotent: a
    prefix already shaped ``<dir>/<x>/<x>`` is returned unchanged so a
    re-run does not nest twice.
    """
    s = (out_prefix or "").strip()
    if not s:
        return s
    p = Path(s)
    if p.parent.name == p.name:  # already <dir>/<stem>/<stem>
        return s
    return str(p / p.name)


def split_person_name(person: str) -> tuple[str, str]:
    """Split a free-text name on whitespace: first token → given name,
    remainder → family name."""
    parts = (person or "").strip().split()
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def validate(form: Mapping[str, str]) -> str | None:
    """Return None when the form is acceptable, else a human-readable error.

    Mirrors run_scan.py's prerequisites: capture folder must exist on
    disk; person name + output prefix non-empty; at least one export
    artifact selected; numeric calibration fields positive when filled.
    """
    capture = (form.get("capture") or "").strip()
    person = (form.get("person") or "").strip()
    out_prefix = (form.get("out_prefix") or "").strip()

    if not capture:
        return "Pick a Stray capture folder."
    if not Path(capture).is_dir():
        return f"Capture folder does not exist: {capture}"
    if not person:
        return "Enter a person name."
    if not out_prefix:
        return "Output prefix is empty."
    if not (form.get("csv") or form.get("obj") or form.get("smis")):
        return "Pick at least one export artifact."

    gender = (form.get("gender") or "female").strip()
    if gender not in VALID_GENDERS:
        return f"Unknown gender: {gender!r}"
    if gender not in ENABLED_GENDERS:
        return (f"Gender {gender!r} is not currently supported "
                "(no SMPL-X model file present).")

    bday = (form.get("birthday") or "").strip()
    if bday and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", bday):
        return f"Birthday must be yyyy-mm-dd: {bday!r}"

    # Height + waist height + girths, when filled, must be positive numbers.
    if (form.get("height") or "").strip() and _parse_pos_float(
            form.get("height")) is None:
        return f"Height must be a positive number (cm): {form.get('height')!r}"
    if (form.get("waist_height") or "").strip() and _parse_pos_float(
            form.get("waist_height")) is None:
        return ("Waist height must be a positive number (cm): "
                f"{form.get('waist_height')!r}")
    wh = _parse_pos_float(form.get("waist_height"))
    h = _parse_pos_float(form.get("height"))
    if wh is not None and h is not None and wh >= h:
        return (f"Waist height ({wh:g} cm) must be smaller than "
                f"height ({h:g} cm).")
    for field, _code, label in TAPE_GIRTHS:
        raw = (form.get(field) or "").strip()
        if raw and _parse_pos_float(raw) is None:
            return f"{label} must be a positive number (cm): {raw!r}"
    return None


def validate_capture_only(form: Mapping[str, str]) -> str | None:
    """Lightweight check for the preflight action: just a real capture dir.

    Preflight only reads the capture, so it doesn't need the person /
    output / export fields that :func:`validate` requires for a full run.
    """
    capture = (form.get("capture") or "").strip()
    if not capture:
        return "Pick a Stray capture folder."
    if not Path(capture).is_dir():
        return f"Capture folder does not exist: {capture}"
    return None


def build_preflight_cmd(form: Mapping[str, str]) -> list[str]:
    """``python -m tailor_twin.preflight <capture>`` argv for the GUI's
    Check-capture button. Streams the same depth/drift verdict as the CLI."""
    capture = (form.get("capture") or "").strip()
    return [PIPELINE_PY, "-m", "tailor_twin.preflight", capture]


def build_cmd(form: Mapping[str, str]) -> list[str]:
    """Translate the validated form into the run_scan.py argv list.

    Caller is expected to have run :func:`validate` first; this function
    trusts the input and emits flags in a stable order so tests can
    assert exact argv content.
    """
    capture = (form.get("capture") or "").strip()
    out_prefix = nest_out_prefix(form.get("out_prefix") or "")
    given, family = split_person_name(form.get("person") or "")
    csv_flag = "--export-csv" if form.get("csv") else "--no-export-csv"
    obj_flag = "--export-obj" if form.get("obj") else "--no-export-obj"
    smis_flag = "--export-smis" if form.get("smis") else "--no-export-smis"

    gender = (form.get("gender") or "female").strip()
    cmd: list[str] = [
        PIPELINE_PY, *RUN_SCAN_ARGS, capture,
        "--out-prefix", out_prefix,
        "--gender", gender,
        csv_flag, obj_flag, smis_flag,
    ]

    # Height anchor (scale).
    height = _parse_pos_float(form.get("height"))
    if height is not None:
        cmd.extend(["--height", f"{height:g}"])

    # Waist height anchor (floor → natural waist, pins the waist line Y).
    waist_height = _parse_pos_float(form.get("waist_height"))
    if waist_height is not None:
        cmd.extend(["--waist-height", f"{waist_height:g}"])

    # Tape girth anchors → one --tape-anchor CODE=cm per filled field.
    for field, code, _label in TAPE_GIRTHS:
        val = _parse_pos_float(form.get(field))
        if val is not None:
            cmd.extend(["--tape-anchor", f"{code}={val:g}"])

    # Landmark vertex-id fixes: "name=vid" tokens, space/comma separated.
    for tok in (form.get("landmark_fix") or "").replace(",", " ").split():
        if "=" in tok:
            cmd.extend(["--landmark-vid", tok])

    # Drift-corrected fusion (opt-in checkbox).
    if form.get("pose_graph"):
        cmd.append("--pose-graph")
    # EXPERIMENTAL fusion-free fit (ROADMAP B1): skip TSDF, fit SMPL-X
    # straight to the multi-frame point cloud.
    if form.get("no_fusion"):
        cmd.append("--no-fusion")
    # Clean-fit (symmetrize + head/hand + A-pose) is on by default in the
    # pipeline; emit the opt-out only when the box is explicitly unchecked.
    if "clean_fit" in form and not form.get("clean_fit"):
        cmd.append("--no-clean-fit")

    bday = (form.get("birthday") or "").strip()
    if bday:
        cmd.extend(["--person-birth-date", bday])
    if given:
        cmd.extend(["--person-given-name", given])
    if family:
        cmd.extend(["--person-family-name", family])
    return cmd
