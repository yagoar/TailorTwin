"""Flask app factory + route handlers for the body-scan GUI."""
from __future__ import annotations

import datetime as dt
import json
import queue
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
)

from .config import (
    DEFAULT_CAPTURES_DIR,
    DEFAULT_RESULTS_DIR,
    GENDERS,
    REPO_ROOT,
    TAPE_GIRTHS,
    WAIST_COLORS,
)
from .forms import build_cmd, validate
from .runner import Runner
from .viewer_data import list_scans, scan_payload


def create_app(runner: Runner | None = None) -> Flask:
    """Build the Flask app. Pass ``runner`` for tests; otherwise a fresh
    :class:`Runner` rooted at the repo is created."""
    app = Flask(__name__)  # uses templates/ and static/ next to this module.
    app.config["RUNNER"] = runner or Runner(cwd=REPO_ROOT)

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            today=dt.date.today().isoformat(),
            colors=WAIST_COLORS,
            genders=GENDERS,
            girths=TAPE_GIRTHS,
            default_captures=str(DEFAULT_CAPTURES_DIR),
            default_results=str(DEFAULT_RESULTS_DIR),
        )

    @app.post("/run")
    def run() -> Any:
        form = request.form.to_dict(flat=True)
        err = validate(form)
        if err:
            return jsonify(ok=False, error=err)
        cmd = build_cmd(form)
        try:
            app.config["RUNNER"].start(cmd)
        except RuntimeError as e:
            return jsonify(ok=False, error=str(e))
        return jsonify(ok=True)

    @app.post("/cancel")
    def cancel() -> Any:
        app.config["RUNNER"].cancel()
        return jsonify(ok=True)

    @app.get("/stream")
    def stream() -> Response:
        runner = app.config["RUNNER"]

        def gen():
            while True:
                try:
                    msg = runner.q.get(timeout=0.5)
                except queue.Empty:
                    # Keep connection alive while waiting for output.
                    if not runner.is_running():
                        yield f"data: {json.dumps({'done': True, 'rc': -1})}\n\n"
                        return
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("done"):
                    return

        return Response(
            gen(), mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # -----------------------------------------------------------------
    # 3D viewer
    # -----------------------------------------------------------------

    @app.get("/viewer")
    def viewer() -> str:
        return render_template(
            "viewer.html",
            today=dt.date.today().isoformat(),
            default_results=str(DEFAULT_RESULTS_DIR),
        )

    def _resolve_dir() -> Path:
        """Pick the results directory: ?dir= query param if provided
        and a real dir, else the project default."""
        raw = (request.args.get("dir") or "").strip()
        if not raw:
            return DEFAULT_RESULTS_DIR
        p = Path(raw).expanduser().resolve()
        if not p.is_dir():
            abort(400, description=f"not a directory: {raw}")
        return p

    @app.get("/api/scans")
    def api_scans() -> Any:
        d = _resolve_dir()
        return jsonify(dir=str(d), scans=list_scans(d))

    @app.get("/api/scan/<name>")
    def api_scan(name: str) -> Any:
        d = _resolve_dir()
        try:
            payload = scan_payload(d, name)
        except FileNotFoundError as e:
            abort(404, description=str(e))
        return jsonify(payload)

    @app.get("/api/scan/<name>/obj")
    def api_scan_obj(name: str) -> Any:
        d = _resolve_dir()
        obj = (d / f"{name}_fit_body.obj").resolve()
        if not obj.is_file() or d.resolve() not in obj.parents:
            abort(404)
        return send_file(obj, mimetype="text/plain")

    @app.get("/api/scan/<name>/bent-arm-obj")
    def api_scan_bent_arm_obj(name: str) -> Any:
        d = _resolve_dir()
        obj = (d / f"{name}_bent_arm_body.obj").resolve()
        if not obj.is_file() or d.resolve() not in obj.parents:
            abort(404)
        return send_file(obj, mimetype="text/plain")

    return app
