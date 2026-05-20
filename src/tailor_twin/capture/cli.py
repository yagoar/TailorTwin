"""CLI: ``tailor-twin capture`` — run the phone capture webapp.

Starts an HTTPS server on the LAN and prints a URL to open on the phone.
The phone's gyroscope gates the shutter so each photo is taken with the
device held vertical (near-orthographic projection). Front + side photos
are saved ready for ``silhouette-fit``.

Example::

    tailor-twin capture --out-dir data/captures/me_photos
    # open the printed https://<lan-ip>:8443/ URL on the phone
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_dir = Path("data/captures") / (
        "photos_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--out-dir", type=Path, default=default_dir,
                   help="Where front.jpg / side.jpg are written.")
    p.add_argument("--host", default="0.0.0.0",
                   help="Bind address (0.0.0.0 = reachable on the LAN).")
    p.add_argument("--port", type=int, default=8443)
    args = p.parse_args(argv)

    from .server import serve
    return serve(args.out_dir, host=args.host, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())
