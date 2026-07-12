"""Unit tests for the measurement history store (repeatability tracking)."""
from __future__ import annotations

from pathlib import Path

from tailor_twin.history import (
    drift_rows,
    history_db_for,
    previous_values,
    record_run,
)


def test_first_run_has_no_previous(tmp_path: Path) -> None:
    db = tmp_path / "history.sqlite"
    assert previous_values(db, person="Yaiza") is None  # DB doesn't exist yet
    record_run(db, person="Yaiza", out_prefix="r1", values={"G07": 69.0})
    assert previous_values(db, person="Someone Else") is None


def test_record_and_fetch_latest(tmp_path: Path) -> None:
    db = tmp_path / "history.sqlite"
    record_run(db, person="Yaiza", out_prefix="r1",
               values={"G07": 69.0, "G04": 87.5})
    record_run(db, person="Yaiza", out_prefix="r2",
               values={"G07": 69.4, "G04": 87.6})
    prev = previous_values(db, person="Yaiza")
    assert prev == {"G07": 69.4, "G04": 87.6}  # latest run wins


def test_history_is_per_person(tmp_path: Path) -> None:
    db = tmp_path / "history.sqlite"
    record_run(db, person="Yaiza", out_prefix="r1", values={"G07": 69.0})
    record_run(db, person="Carmen", out_prefix="r2", values={"G07": 75.0})
    assert previous_values(db, person="Yaiza") == {"G07": 69.0}
    assert previous_values(db, person="Carmen") == {"G07": 75.0}


def test_meta_roundtrip_does_not_break_insert(tmp_path: Path) -> None:
    db = tmp_path / "history.sqlite"
    run_id = record_run(db, person="Yaiza", out_prefix="r1",
                        values={"A01": 160.0},
                        meta={"tape_anchors": {"G04": 87.5},
                              "waist_height_cm": 100.0})
    assert run_id == 1


def test_drift_rows_thresholds_and_sorting() -> None:
    prev = {"G07": 69.0, "G04": 87.5, "G09": 99.0, "ONLY_PREV": 1.0}
    curr = {"G07": 70.5, "G04": 87.6, "G09": 96.0, "ONLY_CURR": 2.0}
    rows = drift_rows(curr, prev, tol_cm=1.0)
    # G04 moved 0.1 (below tol); codes present in only one run are skipped.
    assert [r[0] for r in rows] == ["G09", "G07"]  # sorted by |delta| desc
    code, p, c, d = rows[0]
    assert (p, c) == (99.0, 96.0)
    assert d == -3.0


def test_history_db_location_flat_and_nested(tmp_path: Path) -> None:
    # GUI-nested prefix <results>/<stem>/<stem> → DB in <results>/.
    nested = tmp_path / "results" / "yaiza_x" / "yaiza_x"
    assert history_db_for(nested) == tmp_path / "results" / "history.sqlite"
    # Flat prefix <results>/<stem> → same place.
    flat = tmp_path / "results" / "yaiza_x"
    assert history_db_for(flat) == tmp_path / "results" / "history.sqlite"
