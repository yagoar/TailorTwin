"""Typer entry point for the ``tailor-twin`` console script.

Exposes three subcommands:

* ``scan``      — end-to-end pipeline (Stray → mesh → fit → measurements)
* ``gui``       — launch the localhost web front-end
* ``preflight`` — pre-scan capture sanity check

Each subcommand forwards extra arguments to the underlying module's
``main(argv)`` function, so all native ``argparse`` flags work as before
(e.g. ``tailor-twin scan ./capture --out-prefix data/results/x --voxel 0.005``).

Build artefacts: ``pyproject.toml`` declares
``[project.scripts] tailor-twin = "tailor_twin.cli:app"`` so a
``pip install -e .`` produces the ``tailor-twin`` executable on PATH.
"""
from __future__ import annotations

import typer

app = typer.Typer(
    help="TailorTwin — 3D body scan → measurements → patterns.",
    add_completion=False,
    no_args_is_help=True,
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": ["-h", "--help"],
    },
)


@app.command(
    name="scan",
    help="Run the full pipeline on a Stray Scanner capture.",
    context_settings={"allow_extra_args": True,
                       "ignore_unknown_options": True},
)
def scan_cmd(ctx: typer.Context) -> None:
    """Forward to tailor_twin.scan.main(argv)."""
    from tailor_twin.scan import main as scan_main
    raise typer.Exit(scan_main(ctx.args))


@app.command(
    name="gui",
    help="Launch the localhost web GUI (Flask + Three.js viewer).",
)
def gui_cmd(
    port: int = typer.Option(8060, help="Port to bind."),
    host: str = typer.Option("127.0.0.1", help="Host to bind."),
    no_open: bool = typer.Option(
        False, "--no-open",
        help="Don't auto-open a browser tab.",
    ),
) -> None:
    from tailor_twin.gui import serve
    raise typer.Exit(serve(host=host, port=port, open_browser=not no_open))


@app.command(
    name="preflight",
    help="Inspect a Stray capture: frame count, drift, depth quality.",
    context_settings={"allow_extra_args": True,
                       "ignore_unknown_options": True},
)
def preflight_cmd(ctx: typer.Context) -> None:
    from tailor_twin.preflight import main as preflight_main
    raise typer.Exit(preflight_main(ctx.args))


@app.command(
    name="bent-arm",
    help="Re-pose a fit npz with the elbow flexed and dump L01/L02/L04.",
    context_settings={"allow_extra_args": True,
                       "ignore_unknown_options": True},
)
def bent_arm_cmd(ctx: typer.Context) -> None:
    from tailor_twin.measure.extract_bent_arm import main as bent_arm_main
    raise typer.Exit(bent_arm_main(ctx.args))


@app.command(
    name="refine-tape",
    help="Adjust a fit's betas so tape-measured values (height/bust/"
         "waist/hip/…) match the user's real measurements.",
    context_settings={"allow_extra_args": True,
                       "ignore_unknown_options": True},
)
def refine_tape_cmd(ctx: typer.Context) -> None:
    from tailor_twin.fit.refine_to_tape_cli import main as refine_main
    raise typer.Exit(refine_main(ctx.args))


@app.command(
    name="shapy-import",
    help="Import SHAPY regressor output(s) (image → SMPL-X betas) as a "
         "tailor-twin fit_npz. Photos-only front-end: no LiDAR scan.",
    context_settings={"allow_extra_args": True,
                       "ignore_unknown_options": True},
)
def shapy_import_cmd(ctx: typer.Context) -> None:
    from tailor_twin.io.shapy_import_cli import main as shapy_import_main
    raise typer.Exit(shapy_import_main(ctx.args))


@app.command(
    name="ring-deform",
    help="Deform a fit's mesh per-ring so PlanarGirth circumferences "
         "(bust/waist/hip/…) hit tape exactly. Geometric edit — no "
         "shape-space coupling, unlike refine-tape.",
    context_settings={"allow_extra_args": True,
                       "ignore_unknown_options": True},
)
def ring_deform_cmd(ctx: typer.Context) -> None:
    from tailor_twin.fit.ring_deform_cli import main as ring_deform_main
    raise typer.Exit(ring_deform_main(ctx.args))


@app.command(
    name="fuse-priors",
    help="Combine a LiDAR-chamfer fit and a SHAPY image-regression fit "
         "into one SMPL-X fit npz. Best-of-both: LiDAR's scan-specific "
         "high-order betas + SHAPY's CAESAR-trained anthropometric prior "
         "on the dominant shape modes.",
    context_settings={"allow_extra_args": True,
                       "ignore_unknown_options": True},
)
def fuse_priors_cmd(ctx: typer.Context) -> None:
    from tailor_twin.fit.fuse_priors_cli import main as fuse_priors_main
    raise typer.Exit(fuse_priors_main(ctx.args))


if __name__ == "__main__":
    app()
