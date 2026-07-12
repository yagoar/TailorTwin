"""Static config for the GUI: choices, paths, Python interpreter selection.

Kept dependency-free so it can be imported from tests without needing
Flask or the tailor-twin pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo root is two levels above this file: <root>/src/tailor_twin/gui/.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "data" / "results"
DEFAULT_CAPTURES_DIR = REPO_ROOT / "data" / "captures"

# The GUI may run under any Python that has Flask + Jinja, but the
# pipeline subprocess needs the project's venv (torch, smplx, open3d,
# trimesh, …). Prefer the venv's interpreter when present.
_VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"
PIPELINE_PY: str = str(_VENV_PY) if _VENV_PY.is_file() else sys.executable

# Pipeline entrypoint = ``python -m tailor_twin.scan``. Kept as a list
# so :func:`build_cmd` can prepend it to argv without re-deriving on
# every request.
RUN_SCAN_ARGS: tuple[str, ...] = ("-m", "tailor_twin.scan")

# Optional tape-measured girths the GUI can collect to calibrate the fit
# exactly (passed to the pipeline as --tape-anchor CODE=cm). Each tuple is
# (form field name, seamly code, human label). Torso girths scale about the
# torso centreline; the leg girths (M*) scale per-leg.
TAPE_GIRTHS: tuple[tuple[str, str, str], ...] = (
    ("highbust",  "G03", "High bust"),
    ("bust",      "G04", "Bust"),
    ("underbust", "G05", "Underbust"),
    ("waist",     "G07", "Waist"),
    ("highhip",   "G08", "High hip"),
    ("hip",       "G09", "Hip"),
    ("thigh",     "M03", "Thigh"),
    ("knee",      "M05", "Knee"),
    ("calf",      "M07", "Calf"),
    ("ankle",     "M09", "Ankle"),
)

# Gender values match SeamlyMe (Seamly2D/src/libs/vformat/measurements.cpp):
# {"male", "female", "unknown"}. The HTML lists all three; availability
# is derived from which SMPL-X model files actually sit under
# data/body_models/smplx/. Anything that's never produced a body has to
# go to "Unknown" in SeamlyMe terms, so we keep "neutral" mapped to
# SMPLX_NEUTRAL.npz but label it as "Unknown / Neutral" to match
# SeamlyMe's wording.
_SMPLX_DIR = REPO_ROOT / "data" / "body_models" / "smplx"
_GENDER_MODEL_FILE: dict[str, str] = {
    "female":  "SMPLX_FEMALE.npz",
    "male":    "SMPLX_MALE.npz",
    "neutral": "SMPLX_NEUTRAL.npz",
}


def _gender_enabled(gender: str) -> bool:
    fname = _GENDER_MODEL_FILE.get(gender)
    return bool(fname and (_SMPLX_DIR / fname).is_file())


def _gender_label(gender: str, base: str) -> str:
    if _gender_enabled(gender):
        return base
    return f"{base} — SMPL-X model not present"


GENDERS: tuple[tuple[str, str, bool], ...] = tuple(
    (g, _gender_label(g, lbl), _gender_enabled(g))
    for g, lbl in (
        ("female",  "Female"),
        ("male",    "Male"),
        ("neutral", "Unknown / Neutral"),
    )
)
VALID_GENDERS: frozenset[str] = frozenset(v for v, _, _ in GENDERS)
ENABLED_GENDERS: frozenset[str] = frozenset(v for v, _, e in GENDERS if e)
