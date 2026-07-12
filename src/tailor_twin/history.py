"""Measurement history — SQLite store for run-over-run repeatability.

For a personal tool the most valuable accuracy metric is repeatability:
does the same body, scanned again, produce the same numbers? Every scan
run records its Seamly catalog values here, and the pipeline prints the
codes that drifted versus the same person's previous run so a capture
problem (moved subject, loose clothing, bad anchor) surfaces immediately
instead of as a garment that doesn't fit.

Stdlib-only (sqlite3 + json) so it imports without the ML stack and is
unit-testable anywhere. The DB lives next to the results tree — one file
across all runs — and a failure to record must never fail a scan (the
caller wraps this in try/except).
"""
from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from pathlib import Path


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,          -- ISO-8601 UTC
    person     TEXT NOT NULL,
    out_prefix TEXT NOT NULL,
    meta       TEXT                    -- JSON blob (free-form)
);
CREATE TABLE IF NOT EXISTS measurements (
    run_id   INTEGER NOT NULL REFERENCES runs(id),
    code     TEXT    NOT NULL,
    value_cm REAL    NOT NULL,
    PRIMARY KEY (run_id, code)
);
"""


def history_db_for(out_prefix: Path) -> Path:
    """DB path for a run's ``--out-prefix``: one file per results tree.

    The GUI nests runs as ``<results>/<stem>/<stem>`` (see
    ``gui.forms.nest_out_prefix``); the DB must sit at ``<results>/``
    so every run shares it. A flat ``<results>/<stem>`` prefix keeps the
    DB in ``<results>/`` too.
    """
    p = Path(out_prefix)
    root = p.parent.parent if p.parent.name == p.name else p.parent
    return root / "history.sqlite"


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def record_run(
    db_path: Path,
    *,
    person: str,
    out_prefix: str,
    values: dict[str, float],
    meta: dict | None = None,
) -> int:
    """Insert one run + its measurements; returns the new run id."""
    ts = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO runs (ts, person, out_prefix, meta) VALUES (?,?,?,?)",
            (ts, person, out_prefix,
             json.dumps(meta) if meta is not None else None))
        run_id = int(cur.lastrowid)
        conn.executemany(
            "INSERT INTO measurements (run_id, code, value_cm) VALUES (?,?,?)",
            [(run_id, str(c), float(v)) for c, v in values.items()])
    return run_id


def previous_values(db_path: Path, *, person: str) -> dict[str, float] | None:
    """The most recent recorded run for ``person``, or None if first run."""
    if not Path(db_path).is_file():
        return None
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT id FROM runs WHERE person = ? ORDER BY id DESC LIMIT 1",
            (person,)).fetchone()
        if row is None:
            return None
        rows = conn.execute(
            "SELECT code, value_cm FROM measurements WHERE run_id = ?",
            (row[0],)).fetchall()
    return {code: float(v) for code, v in rows}


def drift_rows(
    current: dict[str, float],
    previous: dict[str, float],
    tol_cm: float = 1.0,
) -> list[tuple[str, float, float, float]]:
    """Codes whose value moved ≥ ``tol_cm`` between two runs.

    Returns ``(code, previous_cm, current_cm, delta_cm)`` tuples sorted by
    descending |delta|. Codes present in only one run are skipped — a
    newly extracted or newly skipped code is not drift.
    """
    rows: list[tuple[str, float, float, float]] = []
    for code in sorted(set(current) & set(previous)):
        c = float(current[code])
        p = float(previous[code])
        d = c - p
        if abs(d) >= tol_cm:
            rows.append((code, p, c, d))
    rows.sort(key=lambda r: -abs(r[3]))
    return rows
