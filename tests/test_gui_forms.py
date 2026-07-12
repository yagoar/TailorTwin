"""Unit tests for ``tailor_twin.gui.forms`` and the Runner state machine."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from tailor_twin.gui.config import PIPELINE_PY, RUN_SCAN_ARGS
from tailor_twin.gui.forms import (
    build_cmd,
    build_preflight_cmd,
    nest_out_prefix,
    split_person_name,
    validate,
    validate_capture_only,
)
from tailor_twin.gui.runner import Runner


# ---------------------------------------------------------------------------
# split_person_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("person, expected", [
    ("", ("", "")),
    ("Yaiza", ("Yaiza", "")),
    ("Yaiza Gomez", ("Yaiza", "Gomez")),
    ("Yaiza Maria Gomez Perez", ("Yaiza", "Maria Gomez Perez")),
    ("   Yaiza   Gomez   ", ("Yaiza", "Gomez")),
])
def test_split_person_name(person: str, expected: tuple[str, str]) -> None:
    assert split_person_name(person) == expected


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _good(tmp_path: Path, **overrides) -> dict[str, str]:
    base: dict[str, str] = {
        "capture": str(tmp_path),
        "person": "Yaiza",
        "out_prefix": str(tmp_path / "yaiza_20260517"),
        "csv": "on",
        "obj": "on",
        "smis": "on",
        "birthday": "1990-05-17",
        "scan_date": "2026-05-17",
        "gender": "female",
    }
    base.update(overrides)
    return base


def test_validate_happy_path(tmp_path: Path) -> None:
    assert validate(_good(tmp_path)) is None


def test_validate_missing_capture(tmp_path: Path) -> None:
    err = validate(_good(tmp_path, capture=""))
    assert err and "capture folder" in err.lower()


def test_validate_nonexistent_capture(tmp_path: Path) -> None:
    err = validate(_good(tmp_path, capture=str(tmp_path / "nope")))
    assert err and "does not exist" in err


def test_validate_missing_person(tmp_path: Path) -> None:
    err = validate(_good(tmp_path, person=""))
    assert err and "person" in err.lower()


def test_validate_no_export(tmp_path: Path) -> None:
    err = validate(_good(tmp_path, csv="", obj="", smis=""))
    assert err and "export" in err.lower()


def test_validate_bad_birthday(tmp_path: Path) -> None:
    err = validate(_good(tmp_path, birthday="17/05/1990"))
    assert err and "yyyy-mm-dd" in err.lower()


def test_validate_empty_birthday_is_ok(tmp_path: Path) -> None:
    assert validate(_good(tmp_path, birthday="")) is None


def test_validate_unknown_gender(tmp_path: Path) -> None:
    err = validate(_good(tmp_path, gender="other"))
    assert err and "gender" in err.lower()


def test_validate_female_gender_ok(tmp_path: Path) -> None:
    assert validate(_good(tmp_path, gender="female")) is None


def test_validate_enabled_genders_pass(tmp_path: Path) -> None:
    from tailor_twin.gui.config import ENABLED_GENDERS
    for g in ENABLED_GENDERS:
        assert validate(_good(tmp_path, gender=g)) is None, (
            f"enabled gender {g!r} should validate")


# ---------------------------------------------------------------------------
# build_cmd
# ---------------------------------------------------------------------------


def test_build_cmd_minimal(tmp_path: Path) -> None:
    cmd = build_cmd(_good(
        tmp_path, birthday="", person="Yaiza",
        csv="on", obj="", smis="",
    ))
    assert cmd[0] == PIPELINE_PY
    # ``python -m tailor_twin.scan`` flag pair, then the capture folder.
    assert tuple(cmd[1:3]) == RUN_SCAN_ARGS
    assert cmd[3] == str(tmp_path)
    assert "--out-prefix" in cmd
    assert "--pattern-system" not in cmd
    assert "--export-csv" in cmd
    assert "--no-export-obj" in cmd
    assert "--no-export-smis" in cmd
    assert "--person-birth-date" not in cmd
    assert "--person-given-name" in cmd
    assert cmd[cmd.index("--person-given-name") + 1] == "Yaiza"
    assert "--person-family-name" not in cmd  # single-token name


def test_build_cmd_passes_gender(tmp_path: Path) -> None:
    cmd = build_cmd(_good(tmp_path, gender="female"))
    assert cmd[cmd.index("--gender") + 1] == "female"


def test_nest_out_prefix_folds_into_folder(tmp_path: Path) -> None:
    p = str(tmp_path / "yaiza_20260615")
    assert nest_out_prefix(p) == str(Path(p) / "yaiza_20260615")


def test_nest_out_prefix_idempotent(tmp_path: Path) -> None:
    once = nest_out_prefix(str(tmp_path / "run"))
    assert nest_out_prefix(once) == once  # a re-run must not nest twice


def test_nest_out_prefix_blank() -> None:
    assert nest_out_prefix("") == ""
    assert nest_out_prefix("   ") == ""


def test_build_cmd_nests_out_prefix(tmp_path: Path) -> None:
    # The GUI field holds <results>/<stem>; build_cmd folds the run into a
    # <stem>/ folder so every artifact lands together.
    cmd = build_cmd(_good(tmp_path))
    op = cmd[cmd.index("--out-prefix") + 1]
    assert op == str(tmp_path / "yaiza_20260517" / "yaiza_20260517")


def test_build_cmd_full(tmp_path: Path) -> None:
    cmd = build_cmd(_good(
        tmp_path, birthday="1990-05-17",
        person="Yaiza Gomez Perez",
    ))
    assert ["--person-birth-date", "1990-05-17"] == [
        cmd[cmd.index("--person-birth-date")],
        cmd[cmd.index("--person-birth-date") + 1],
    ]
    assert cmd[cmd.index("--person-given-name") + 1] == "Yaiza"
    assert cmd[cmd.index("--person-family-name") + 1] == "Gomez Perez"
    assert "--pattern-system" not in cmd


def test_build_cmd_height_and_girths(tmp_path: Path) -> None:
    cmd = build_cmd(_good(
        tmp_path, height="170", bust="87.5", waist="69", calf="34",
    ))
    assert ["--height", "170"] == [
        cmd[cmd.index("--height")], cmd[cmd.index("--height") + 1]]
    # each filled girth becomes one --tape-anchor CODE=cm
    anchors = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--tape-anchor"]
    assert "G04=87.5" in anchors
    assert "G07=69" in anchors
    assert "M07=34" in anchors  # leg girth
    # blank girths are not emitted
    assert not any(a.startswith("G09=") for a in anchors)


def test_build_cmd_no_calibration_when_blank(tmp_path: Path) -> None:
    cmd = build_cmd(_good(tmp_path))
    assert "--height" not in cmd
    assert "--waist-height" not in cmd
    assert "--tape-anchor" not in cmd


def test_build_cmd_waist_height(tmp_path: Path) -> None:
    cmd = build_cmd(_good(tmp_path, waist_height="100.5"))
    assert ["--waist-height", "100.5"] == [
        cmd[cmd.index("--waist-height")],
        cmd[cmd.index("--waist-height") + 1],
    ]


def test_validate_waist_height_ok(tmp_path: Path) -> None:
    assert validate(_good(tmp_path, waist_height="100")) is None
    assert validate(_good(tmp_path, waist_height="")) is None


def test_validate_bad_waist_height(tmp_path: Path) -> None:
    for bad in ("abc", "-5", "0"):
        err = validate(_good(tmp_path, waist_height=bad))
        assert err and "waist height" in err.lower(), bad


def test_validate_waist_height_exceeds_height(tmp_path: Path) -> None:
    err = validate(_good(tmp_path, height="160", waist_height="170"))
    assert err and "smaller than" in err
    # equal is also rejected; and without a height it can't be checked.
    assert validate(_good(tmp_path, height="160", waist_height="160"))
    assert validate(_good(tmp_path, waist_height="170")) is None


def test_build_cmd_pose_graph_and_clean_fit(tmp_path: Path) -> None:
    cmd = build_cmd(_good(tmp_path, pose_graph="on"))
    assert "--pose-graph" in cmd
    # clean_fit default on (checkbox checked) -> no opt-out flag
    assert "--no-clean-fit" not in build_cmd(_good(tmp_path, clean_fit="on"))
    # clean_fit present-but-unchecked (key absent from POST) -> opt-out.
    # Browsers omit unchecked boxes, so simulate the hidden marker.
    cmd_off = build_cmd(_good(tmp_path, clean_fit=""))
    assert "--no-clean-fit" in cmd_off


def test_validate_bad_height(tmp_path: Path) -> None:
    err = validate(_good(tmp_path, height="-5"))
    assert err and "height" in err.lower()


def test_validate_bad_girth(tmp_path: Path) -> None:
    err = validate(_good(tmp_path, calf="abc"))
    assert err and "calf" in err.lower()


# ---------------------------------------------------------------------------
# preflight (Check capture)
# ---------------------------------------------------------------------------


def test_validate_capture_only_ok(tmp_path: Path) -> None:
    # preflight needs only a real capture dir, not person/output/exports.
    assert validate_capture_only({"capture": str(tmp_path)}) is None


def test_validate_capture_only_missing(tmp_path: Path) -> None:
    assert validate_capture_only({"capture": ""})
    err = validate_capture_only({"capture": str(tmp_path / "nope")})
    assert err and "does not exist" in err


def test_build_preflight_cmd(tmp_path: Path) -> None:
    cmd = build_preflight_cmd({"capture": str(tmp_path)})
    assert cmd[0] == PIPELINE_PY
    assert cmd[1:] == ["-m", "tailor_twin.preflight", str(tmp_path)]


# ---------------------------------------------------------------------------
# Runner state machine
# ---------------------------------------------------------------------------


def _drain(runner: Runner, *, timeout: float = 5.0) -> list[dict]:
    """Collect messages from runner.q until {'done': True} or timeout."""
    deadline = time.monotonic() + timeout
    msgs: list[dict] = []
    while time.monotonic() < deadline:
        try:
            msg = runner.q.get(timeout=0.2)
        except Exception:  # noqa: BLE001 — queue.Empty
            continue
        msgs.append(msg)
        if msg.get("done"):
            return msgs
    raise TimeoutError("runner did not finish in time")


def test_runner_runs_to_completion(tmp_path: Path) -> None:
    runner = Runner(cwd=tmp_path)
    runner.start(["/bin/sh", "-c", "echo hello; echo world"])
    msgs = _drain(runner)
    # First message is the command echo.
    assert msgs[0]["line"].startswith("$ ")
    body = "".join(m["line"] for m in msgs if "line" in m)
    assert "hello" in body and "world" in body
    assert msgs[-1] == {"done": True, "rc": 0}
    assert not runner.is_running()


def test_runner_rejects_double_start(tmp_path: Path) -> None:
    runner = Runner(cwd=tmp_path)
    runner.start(["/bin/sh", "-c", "sleep 0.5"])
    with pytest.raises(RuntimeError, match="already running"):
        runner.start(["/bin/sh", "-c", "echo nope"])
    _drain(runner)  # let the first one finish so no zombie.


def test_runner_cancel_terminates(tmp_path: Path) -> None:
    runner = Runner(cwd=tmp_path)
    runner.start(["/bin/sh", "-c", "sleep 30"])
    assert runner.is_running()
    runner.cancel()
    msgs = _drain(runner, timeout=3.0)
    assert msgs[-1]["done"] is True
    assert msgs[-1]["rc"] != 0
    assert any("cancelled" in m.get("line", "") for m in msgs)


def test_runner_cancel_idle_is_noop(tmp_path: Path) -> None:
    runner = Runner(cwd=tmp_path)
    runner.cancel()  # nothing running; must not raise.
    assert not runner.is_running()
