"""Interactive vertex picker — find the SMPL-X vertex id for a landmark.

Renders a fitted body as a hoverable Plotly point cloud: hover any vertex
to read its id, the current landmark vertex is marked in red. Use it to
hand-correct a mis-placed landmark, then either pass the id to
``--landmark-vid NAME=id`` (scan / measure) or patch
``references/smplx_landmark_review.json``.

    python -m scripts.pick_vertex data/results/NAME_smplx_fit.npz \
        --landmark acromion_left

Opens a browser tab. Rotate to the region, hover the correct anatomical
point, note the "vid N" in the tooltip. A *_left fix auto-mirrors to
*_right in the measure step, so you only need to pick the left side.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

DEFAULT_REVIEW_JSON = Path("references/smplx_landmark_review.json")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("fit_npz", type=Path)
    p.add_argument("--landmark", default="acromion_left",
                   help="Landmark whose current vertex to highlight "
                        "(default acromion_left).")
    p.add_argument("--review-json", type=Path, default=DEFAULT_REVIEW_JSON)
    p.add_argument("--region", default="upper",
                   choices=["upper", "all"],
                   help="'upper' (default) shows the chest-up region (less "
                        "clutter for shoulder/neck picks); 'all' the whole body.")
    args = p.parse_args(argv)

    fit = np.load(args.fit_npz)
    v = fit["smplx_vertices"].astype(np.float64)
    ids = np.arange(len(v))

    cur_vid = None
    if args.review_json.is_file():
        review = json.loads(args.review_json.read_text())
        rec = review.get(args.landmark)
        if isinstance(rec, dict) and "vertex_id" in rec:
            cur_vid = int(rec["vertex_id"])

    sel = np.ones(len(v), dtype=bool)
    if args.region == "upper":
        # Chest-up: keep vertices in the top ~40 % of the body height.
        y = v[:, 1]
        sel = y > (y.min() + 0.58 * (y.max() - y.min()))
        if cur_vid is not None:
            sel[cur_vid] = True

    pts = v[sel]
    pid = ids[sel]
    cloud = go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=dict(size=2, color="lightgray"),
        customdata=pid,
        hovertemplate="vid %{customdata}<br>x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}"
                      "<extra></extra>",
        name="vertices",
    )
    traces = [cloud]
    if cur_vid is not None:
        c = v[cur_vid]
        traces.append(go.Scatter3d(
            x=[c[0]], y=[c[1]], z=[c[2]], mode="markers",
            marker=dict(size=7, color="red"),
            customdata=[cur_vid],
            hovertemplate=f"CURRENT {args.landmark}<br>vid %{{customdata}}"
                          "<extra></extra>",
            name=f"current {args.landmark} ({cur_vid})",
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Hover a vertex to read its id — pick the correct "
              f"{args.landmark}",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.show()
    print(f"current {args.landmark}: vid {cur_vid}")
    print("Hover the correct anatomical point, note 'vid N', then:")
    print(f"  scan/measure: --landmark-vid {args.landmark}=N")
    print(f"  or patch {args.review_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
