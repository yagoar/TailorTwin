"""Flask server for the phone capture webapp.

Serves a single mobile page (``templates/capture.html``) and accepts the
two captured photos on ``POST /upload``. Runs over HTTPS with an adhoc
self-signed certificate because ``getUserMedia`` (camera) and the iOS
``DeviceOrientationEvent`` permission prompt only work in a *secure
context* — and a phone reaching the laptop over the LAN is not
``localhost``, so plain HTTP will not do.

Uploaded payload (multipart form):
  * ``front`` / ``side`` — JPEG blobs
  * ``height_cm`` / ``weight_kg`` / ``gender`` — text fields
  * ``meta``             — JSON: per-shot tilt (pitch/roll) at capture

Saved layout under ``out_dir``::

    <out_dir>/front.jpg
    <out_dir>/side.jpg
    <out_dir>/capture_meta.json
"""
from __future__ import annotations

import json
import socket
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

    @app.route("/upload", methods=["POST"])
    def upload():
        front = request.files.get("front")
        side = request.files.get("side")
        if front is None or side is None:
            return jsonify(ok=False, error="both photos required"), 400
        front.save(out_dir / "front.jpg")
        side.save(out_dir / "side.jpg")
        meta = {
            "captured_at": datetime.now().isoformat(timespec="seconds"),
            "height_cm": request.form.get("height_cm", ""),
            "weight_kg": request.form.get("weight_kg", ""),
            "gender": request.form.get("gender", ""),
        }
        try:
            meta["shots"] = json.loads(request.form.get("meta", "{}"))
        except json.JSONDecodeError:
            meta["shots"] = {}
        (out_dir / "capture_meta.json").write_text(
            json.dumps(meta, indent=2))
        return jsonify(ok=True, saved=str(out_dir))

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
