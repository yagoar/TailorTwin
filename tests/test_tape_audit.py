"""Unit tests for the ring-deform tape audit (unanchored-drift detector)."""
from __future__ import annotations

from tailor_twin.fit.ring_deform import audit_girth_drift


def test_flags_only_unanchored_codes_over_tol() -> None:
    before = {"G04": 90.0, "G07": 72.0, "G06": 66.0, "H05": 40.0}
    after = {"G04": 87.5, "G07": 69.0, "G06": 64.5, "H05": 40.2}
    # G04/G07 anchored (moved on purpose); G06 collateral -1.5; H05 +0.2.
    audit = audit_girth_drift(before, after, anchored={"G04", "G07"})
    assert audit["flagged_unanchored"] == ["G06"]
    assert audit["drift"]["G06"]["delta_cm"] == -1.5
    assert audit["drift"]["G06"]["anchored"] is False
    assert audit["drift"]["G04"]["anchored"] is True


def test_flagged_sorted_by_magnitude() -> None:
    before = {"A": 10.0, "B": 10.0, "C": 10.0}
    after = {"A": 11.2, "B": 13.0, "C": 10.0}
    audit = audit_girth_drift(before, after, anchored=set())
    assert audit["flagged_unanchored"] == ["B", "A"]


def test_untouched_and_missing_codes_ignored() -> None:
    before = {"G07": 69.0, "STABLE": 30.0, "ONLY_BEFORE": 5.0}
    after = {"G07": 69.0, "STABLE": 30.01, "ONLY_AFTER": 6.0}
    audit = audit_girth_drift(before, after, anchored={"G07"})
    assert audit["flagged_unanchored"] == []
    # Sub-0.05 moves on unanchored codes don't clutter the report; anchored
    # codes are always listed.
    assert "STABLE" not in audit["drift"]
    assert "ONLY_BEFORE" not in audit["drift"]
    assert "ONLY_AFTER" not in audit["drift"]
    assert "G07" in audit["drift"]


def test_non_finite_values_ignored() -> None:
    before = {"G07": float("nan"), "G09": 99.0}
    after = {"G07": 70.0, "G09": 97.0}
    audit = audit_girth_drift(before, after, anchored=set())
    assert audit["flagged_unanchored"] == ["G09"]
    assert "G07" not in audit["drift"]


def test_custom_tolerance() -> None:
    before = {"X": 50.0}
    after = {"X": 50.6}
    assert audit_girth_drift(before, after, set())["flagged_unanchored"] == []
    assert audit_girth_drift(
        before, after, set(), tol_cm=0.5)["flagged_unanchored"] == ["X"]
