"""Run a trained BMnet on two silhouettes → 14 measurements (cm).

``predict_measurements`` is the library entry point: it takes two
silhouette arrays (any source — BodyM masks or a tailor-twin Sapiens
segmentation), height and weight, and returns the 14 measurements in
centimetres. The CLI wraps it for two PNG masks.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from . import MEAS_COLS
from .dataset import build_input
from .model import BMnet, Standardizer


@lru_cache(maxsize=2)
def _load(ckpt_path: str, device: str):
    """Load (model, standardizer, img_h, img_w) once per checkpoint."""
    dev = torch.device(device)
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model = BMnet(n_out=len(ck["meas_cols"])).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    std = Standardizer.from_dict(ck["standardizer"])
    return model, std, ck["img_h"], ck["img_w"], dev


def predict_measurements(
    front: np.ndarray, side: np.ndarray, height_cm: float,
    weight_kg: float, *, ckpt: str | Path = "data/results/bmnet.pt",
    device: str = "cpu",
) -> dict[str, float]:
    """Two silhouettes + H/W  →  {measurement_name: cm}.

    ``front`` / ``side`` are 2-D arrays, non-zero = body."""
    model, std, img_h, img_w, dev = _load(str(ckpt), device)
    x = build_input(front, side, height_cm, weight_kg, std,
                    img_h=img_h, img_w=img_w)
    with torch.no_grad():
        z = model(x[None].to(dev)).cpu().numpy()[0]
    cm = std.denorm_meas(z)
    return {name: float(v) for name, v in zip(MEAS_COLS, cm)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("front", type=Path, help="front silhouette PNG")
    p.add_argument("side", type=Path, help="side silhouette PNG")
    p.add_argument("--height", type=float, required=True)
    p.add_argument("--weight", type=float, required=True)
    p.add_argument("--ckpt", type=Path, default=Path("data/results/bmnet.pt"))
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    import cv2
    front = cv2.imread(str(args.front), cv2.IMREAD_GRAYSCALE)
    side = cv2.imread(str(args.side), cv2.IMREAD_GRAYSCALE)
    if front is None or side is None:
        raise SystemExit("could not read one of the silhouette PNGs")

    pred = predict_measurements(front, side, args.height, args.weight,
                                ckpt=args.ckpt, device=args.device)
    print(f"BMnet measurements (cm)  [ckpt {args.ckpt}]")
    for name, v in pred.items():
        print(f"  {name:<20} {v:7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
