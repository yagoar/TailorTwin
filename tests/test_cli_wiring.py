"""Static CLI-wiring parity checks.

The pipeline is a chain of argparse programs: the GUI builds a
``tailor_twin.scan`` argv, and ``scan.py`` shells out to
``measure.cli`` / ``fit.ring_deform_cli`` / ``measure.extract_bent_arm``
with more flags. A typo'd or missing flag anywhere in that chain only
surfaces minutes into a real run on the user's machine — these tests
catch it statically, without importing the heavy modules (scan.py pulls
in Open3D/cv2 at import time, so everything here is AST-based).

Conventions relied on:
- Declared flags = first positional string arg of ``*.add_argument(...)``
  calls that starts with ``--``.
- Emitted flags = string constants starting with ``--`` that appear
  inside *list literals* (``cmd = [...]`` / ``cmd.extend([...])``).
  argparse declarations pass the flag as a direct call argument, never
  inside a list, so the two sets don't bleed into each other.
- ``--no-X`` satisfies a declared ``--X`` (argparse
  ``BooleanOptionalAction`` derives the negative form).
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "tailor_twin"

SCAN = SRC / "scan.py"
MEASURE_CLI = SRC / "measure" / "cli.py"
RING_CLI = SRC / "fit" / "ring_deform_cli.py"
REFINE_CLI = SRC / "fit" / "refine_to_tape_cli.py"
BENT_ARM = SRC / "measure" / "extract_bent_arm.py"
PREFLIGHT = SRC / "preflight.py"


def declared_flags(path: Path) -> set[str]:
    """--flags declared via add_argument in a module (AST, no import)."""
    tree = ast.parse(path.read_text(), str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for arg in node.args:
                if (isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                        and arg.value.startswith("--")):
                    out.add(arg.value)
    return out


def emitted_flags(path: Path) -> set[str]:
    """--flags appearing inside list literals (subprocess argv pieces)."""
    tree = ast.parse(path.read_text(), str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.List):
            for el in node.elts:
                if (isinstance(el, ast.Constant)
                        and isinstance(el.value, str)
                        and el.value.startswith("--")):
                    out.add(el.value)
    return out


def _normalize(flag: str) -> str:
    """--no-X → --X (BooleanOptionalAction negative form)."""
    return "--" + flag[len("--no-"):] if flag.startswith("--no-") else flag


def _assert_subset(emitted: set[str], declared: set[str], label: str) -> None:
    missing = sorted(f for f in emitted
                     if _normalize(f) not in declared and f not in declared)
    assert not missing, (
        f"{label}: emitted flag(s) with no matching add_argument "
        f"declaration: {missing}")


def test_scan_subprocess_flags_exist_in_target_parsers() -> None:
    # scan.py drives three subprocesses; its emitted flags must exist in
    # the union of their parsers (union is fine — a typo'd flag matches
    # none of them, which is the failure this guards against).
    targets = (declared_flags(MEASURE_CLI) | declared_flags(RING_CLI)
               | declared_flags(BENT_ARM))
    _assert_subset(emitted_flags(SCAN), targets, "scan.py")


def test_calibrator_clis_forward_valid_measure_flags() -> None:
    measure = declared_flags(MEASURE_CLI)
    _assert_subset(emitted_flags(RING_CLI), measure | declared_flags(RING_CLI),
                   "ring_deform_cli.py")
    _assert_subset(emitted_flags(REFINE_CLI), measure,
                   "refine_to_tape_cli.py")


def test_gui_build_cmd_flags_exist_in_scan_parser(tmp_path: Path) -> None:
    # forms.py is light enough to import — exercise build_cmd with every
    # optional field filled so all conditional flags are emitted.
    from tailor_twin.gui.forms import build_cmd, build_preflight_cmd

    form = {
        "capture": str(tmp_path),
        "person": "Yaiza Gomez",
        "out_prefix": str(tmp_path / "x"),
        "csv": "on", "obj": "", "smis": "",
        "gender": "female",
        "birthday": "1990-05-17",
        "height": "160",
        "waist_height": "100",
        "landmark_fix": "acromion_left=4447",
        "pose_graph": "on",
        "clean_fit": "",           # explicit opt-out → --no-clean-fit
        "no_fusion": "on",         # experimental fusion-free fit
        "bust": "87.5", "waist": "69", "hip": "99",
        "highbust": "1", "underbust": "1", "highhip": "1",
        "thigh": "1", "knee": "1", "calf": "1", "ankle": "1",
    }
    scan_declared = declared_flags(SCAN)
    cmd = build_cmd(form)
    gui_flags = {tok for tok in cmd if tok.startswith("--")}
    _assert_subset(gui_flags, scan_declared, "gui build_cmd")
    # Sanity: the interesting conditional flags actually got emitted, so
    # this test can't silently pass on an empty set.
    for expected in ("--height", "--waist-height", "--tape-anchor",
                     "--landmark-vid", "--pose-graph", "--no-clean-fit"):
        assert expected in gui_flags, f"build_cmd stopped emitting {expected}"

    pf = build_preflight_cmd(form)
    assert pf[1:3] == ["-m", "tailor_twin.preflight"]
    assert PREFLIGHT.is_file()


def test_scan_parser_declares_session_critical_flags() -> None:
    # The flags recent work depends on — a rename here must be caught.
    declared = declared_flags(SCAN)
    for flag in ("--waist-height", "--fusion", "--height", "--tape-anchor",
                 "--landmark-vid", "--skip-fusion", "--clean-fit"):
        assert flag in declared, f"scan.py no longer declares {flag}"
    for flag in ("--waist-height-cm", "--waist-y"):
        assert flag in declared_flags(MEASURE_CLI), (
            f"measure/cli.py no longer declares {flag}")
    for flag in ("--waist-height-cm", "--audit", "--target"):
        assert flag in declared_flags(RING_CLI), (
            f"ring_deform_cli.py no longer declares {flag}")
