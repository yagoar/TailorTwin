"""Run manifest — provenance for every pipeline run.

Each ``tailor-twin scan`` writes ``<out-prefix>_manifest.json`` recording
the full configuration the run actually used (every CLI arg), the code
version (git commit), timestamps, and the exit code. Six months later,
when two runs of the same capture disagree, the manifest answers "what
was different?" without archaeology through shell history.

Stdlib-only so it imports without the ML stack and is unit-testable
anywhere. Writing a manifest is best-effort: a failure here must never
fail the scan (the caller wraps this in try/except).
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path


def git_commit(repo_root: Path | None = None) -> str | None:
    """HEAD commit of the repo this module lives in, or None (no git,
    tarball install, …)."""
    root = repo_root or Path(__file__).resolve().parents[2]
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=root)
    except Exception:  # noqa: BLE001 — git missing entirely
        return None
    sha = r.stdout.strip()
    return sha if r.returncode == 0 and sha else None


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def write_manifest(
    out_prefix: Path,
    *,
    config: dict,
    rc: int,
    started: str,
    finished: str,
) -> Path:
    """Write ``<out-prefix>_manifest.json`` and return its path.

    ``config`` is typically ``vars(args)`` from the scan CLI — values that
    aren't JSON-native (Path, tuples, …) are stringified rather than
    rejected, so the manifest never fails on an exotic arg type.
    """
    out_prefix = Path(out_prefix)
    path = out_prefix.with_name(out_prefix.name + "_manifest.json")
    payload = {
        "started": started,
        "finished": finished,
        "exit_code": int(rc),
        "git_commit": git_commit(),
        "config": config,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str, sort_keys=False))
    return path
