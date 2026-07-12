"""tailor_twin — personal LiDAR body-scanning pipeline.

See SPEC.md for the project specification, GUARDRAILS.md for
AI-generation rules, and docs/ROADMAP.md for the improvement plan. The
package is organised into subpackages mirroring the pipeline:

    io/          capture loaders (Stray Scanner)
    preprocess/  segmentation, depth filtering
    reconstruct/ TSDF fusion / multi-frame cloud, mesh cleanup
    fit/         SMPL-X+D fitting, clean-fit, tape-anchor calibration
    measure/     landmarks, recipe primitives, Seamly catalog, exports
    gui/         Flask app + Three.js viewer

Top-level modules: scan.py (end-to-end pipeline), preflight.py
(capture check), history.py (measurement history + drift), manifest.py
(per-run provenance), cli.py (Typer entry).
"""

__version__ = "0.0.1"
