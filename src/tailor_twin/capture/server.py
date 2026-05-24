"""Flask server for the phone capture webapp.

Serves a single mobile page (``templates/capture.html``) and accepts the
two captured photos on ``POST /upload``. Runs over HTTPS with an adhoc
self-signed certificate because ``getUserMedia`` (camera) and the iOS
``DeviceOrientationEvent`` permission prompt only work in a *secure
context* — and a phone reaching the laptop over the LAN is not
``localhost``, so plain HTTP will not do.

Uploaded payload (multipart form):
  * ``front`` / ``back`` / ``left`` / ``right`` — JPEG blobs (360° set)
  * ``height_cm`` / ``weight_kg`` / ``gender`` — text fields
  * ``meta``             — JSON: per-shot tilt (pitch/roll) at capture

Saved layout under ``out_dir``::

    <out_dir>/front.jpg
    <out_dir>/back.jpg
    <out_dir>/left.jpg
    <out_dir>/right.jpg
    <out_dir>/capture_meta.json
"""
from __future__ import annotations

import json
import socket
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request


def lan_ip() -> str:
    """Best-effort primary LAN IPv4 of this machine (no traffic sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # unroutable — just picks the iface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def create_app(out_dir: Path) -> Flask:
    """Build the capture Flask app writing photos into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__)  # uses templates/ next to this module.
    app.config["OUT_DIR"] = out_dir

    @app.route("/")
    def index():
        return render_template("capture.html")

    @app.route("/check", methods=["POST"])
    def check():
        """Sapiens pose-gate a single shot before the phone accepts it.

        Form: ``shot`` (JPEG blob) + ``view`` ("front"|"side"). Returns
        the :func:`check_pose` verdict. If Sapiens is unavailable the
        gate degrades open (``skipped``) so capture still works — the
        gyroscope gate alone then applies."""
        shot = request.files.get("shot")
        view = request.form.get("view", "front")
        if shot is None:
            return jsonify(ok=False, error="no shot"), 400
        import tempfile
        from .pose_check import check_pose
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tf:
            shot.save(tf.name)
            try:
                verdict = check_pose(Path(tf.name), view)
            except SystemExit as e:
                return jsonify(ok=True, skipped=True, reason=str(e))
            except Exception as e:  # noqa: BLE001 — never block capture
                return jsonify(ok=True, skipped=True, reason=repr(e))
        return jsonify(**verdict)

    @app.route("/upload", methods=["POST"])
    def upload():
        views = ("front", "back", "left", "right")
        files = {v: request.files.get(v) for v in views}
        if any(f is None for f in files.values()):
            return jsonify(ok=False, error="all four photos required"), 400
        # Each capture lands in its own timestamped subfolder so repeat
        # sessions (and different subjects) never overwrite each other.
        now = datetime.now()
        dest = out_dir / f"capture_{now:%Y%m%d_%H%M%S}"
        dest.mkdir(parents=True, exist_ok=True)
        for v, f in files.items():
            f.save(dest / f"{v}.jpg")
        meta = {
            "captured_at": now.isoformat(timespec="seconds"),
            "height_cm": request.form.get("height_cm", ""),
            "weight_kg": request.form.get("weight_kg", ""),
            "gender": request.form.get("gender", ""),
        }
        try:
            meta["shots"] = json.loads(request.form.get("meta", "{}"))
        except json.JSONDecodeError:
            meta["shots"] = {}
        (dest / "capture_meta.json").write_text(
            json.dumps(meta, indent=2))
        return jsonify(ok=True, saved=str(dest))

    # ------------------------------------------------------------------
    # Paired 2-phone synchronized capture (front + 90° side at the same
    # instant — no pose drift between views). Two phones open /pair, pick
    # a role, and report "ready" (phone upright). When both are ready the
    # server arms a shared countdown; both phones poll the absolute fire
    # time and grab together. State is in-memory, single session.
    # ------------------------------------------------------------------
    sync = {
        "lock": threading.Lock(),
        "joined": {"front": False, "side": False},
        "ready": {"front": False, "side": False},
        "uploaded": {"front": False, "side": False},
        "fire_at": None,        # epoch seconds, or None
        "dest": None,
        "meta": {},
        "countdown": 6.0,       # seconds from both-ready to capture
    }
    app.config["SYNC"] = sync

    @app.route("/pair")
    def pair():
        return render_template("pair.html")

    @app.route("/sync/join", methods=["POST"])
    def sync_join():
        role = request.form.get("role")
        if role not in ("front", "side"):
            return jsonify(ok=False, error="bad role"), 400
        with sync["lock"]:
            sync["joined"][role] = True
            if role == "front":
                sync["meta"] = {
                    "height_cm": request.form.get("height_cm", ""),
                    "weight_kg": request.form.get("weight_kg", ""),
                    "gender": request.form.get("gender", ""),
                }
        return jsonify(ok=True)

    @app.route("/sync/ready", methods=["POST"])
    def sync_ready():
        role = request.form.get("role")
        if role not in ("front", "side"):
            return jsonify(ok=False), 400
        with sync["lock"]:
            sync["ready"][role] = request.form.get("ready") == "1"
            # Arming is manual (/sync/trigger). But a phone that loses
            # "ready" after a trigger cancels the pending countdown.
            armed_ok = (all(sync["ready"].values())
                        and all(sync["joined"].values()))
            if not armed_ok and sync["fire_at"] is not None \
                    and not any(sync["uploaded"].values()):
                sync["fire_at"] = None
        return jsonify(ok=True)

    @app.route("/sync/trigger", methods=["POST"])
    def sync_trigger():
        """Either phone fires this to start the shared countdown — only
        when both phones have joined and report ready."""
        with sync["lock"]:
            if any(sync["uploaded"].values()):
                return jsonify(ok=False, error="already captured"), 409
            if not (all(sync["ready"].values())
                    and all(sync["joined"].values())):
                return jsonify(ok=False, error="both phones not ready"), 409
            if sync["fire_at"] is None:
                sync["fire_at"] = time.time() + sync["countdown"]
        return jsonify(ok=True)

    @app.route("/sync/state")
    def sync_state():
        with sync["lock"]:
            fa = sync["fire_at"]
            fire_in = int((fa - time.time()) * 1000) if fa else None
            return jsonify(joined=sync["joined"], ready=sync["ready"],
                           uploaded=sync["uploaded"], fire_in_ms=fire_in)

    @app.route("/sync/reset", methods=["POST"])
    def sync_reset():
        with sync["lock"]:
            sync["joined"] = {"front": False, "side": False}
            sync["ready"] = {"front": False, "side": False}
            sync["uploaded"] = {"front": False, "side": False}
            sync["fire_at"] = None
            sync["dest"] = None
            sync["meta"] = {}
        return jsonify(ok=True)

    @app.route("/pair_upload", methods=["POST"])
    def pair_upload():
        role = request.form.get("role")
        shot = request.files.get("shot")
        if role not in ("front", "side") or shot is None:
            return jsonify(ok=False, error="bad upload"), 400
        with sync["lock"]:
            # First upload of the pair creates the timestamped folder.
            if sync["dest"] is None:
                now = datetime.now()
                sync["dest"] = out_dir / f"pair_{now:%Y%m%d_%H%M%S}"
                sync["dest"].mkdir(parents=True, exist_ok=True)
                meta = dict(sync["meta"])
                meta["captured_at"] = now.isoformat(timespec="seconds")
                meta["mode"] = "paired_simultaneous"
                try:
                    meta["shots"] = json.loads(request.form.get("meta", "{}"))
                except json.JSONDecodeError:
                    meta["shots"] = {}
                (sync["dest"] / "capture_meta.json").write_text(
                    json.dumps(meta, indent=2))
            dest = sync["dest"]
            shot.save(dest / f"{role}.jpg")
            sync["uploaded"][role] = True
            done = all(sync["uploaded"].values())
        return jsonify(ok=True, saved=str(dest), done=done)

    return app


def serve(out_dir: Path, host: str = "0.0.0.0", port: int = 8443) -> int:
    """Run the capture server over HTTPS (adhoc self-signed cert).

    Prints the LAN URL to open on the phone. The phone must accept the
    self-signed-certificate browser warning once.
    """
    app = create_app(out_dir)
    url = f"https://{lan_ip()}:{port}/"
    print("\n  TailorTwin capture webapp")
    print(f"  open on your phone:  {url}")
    print("  (accept the self-signed certificate warning once)")
    print(f"  photos will be saved to: {Path(out_dir).resolve()}\n")
    try:
        app.run(host=host, port=port, ssl_context="adhoc", debug=False)
    except (ImportError, TypeError):
        print("HTTPS needs the 'cryptography' package "
              "(pip install cryptography). Camera will not work over "
              "plain HTTP from a phone.")
        return 1
    return 0
