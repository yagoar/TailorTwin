"""Phone capture webapp — gyroscope-gated front + side body photos.

A small self-hosted web page the user opens on their phone. It mirrors
the 3DLook Mobile Tailor capture: the phone's gyroscope (DeviceOrientation
API) gates the shutter so each photo is taken with the phone held truly
vertical — a near-orthographic projection with minimal perspective
distortion, which is what makes silhouette width/depth metric.

Flow: front photo → "turn 90°" → side photo → upload. Photos land in an
output directory ready for ``tailor-twin silhouette-fit`` (after a
Sapiens part-seg pass).

See :func:`tailor_twin.capture.server.create_app`.
"""
from .server import create_app

__all__ = ["create_app"]
