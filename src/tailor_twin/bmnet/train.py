"""Train BMnet on the BodyM dataset.

Follows the paper's recipe: L1 loss on the 14 measurements, Adam, a
multi-step learning-rate schedule, and the BodyM TestA split as the
validation set. The adversarial body simulator (ABS) augmentation is
*not* implemented — it is an optional refinement worth ~10 % in the
paper; plain supervised BMnet already reaches ~1.5 cm waist error,
enough to pin SMPL-X girths.

The loss is evaluated in centimetres: the network emits standardized
measurements, ``|z_pred - z_true| * meas_std`` recovers the cm error,
and that is what is back-propagated and reported.

  python -m tailor_twin.bmnet.train --epochs 80 \
      --data-root data/bodym --out data/results/bmnet.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import MEAS_COLS
from .dataset import BodyMDataset, compute_standardizer
from .model import BMnet


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _evaluate(model, loader, std_t, dev) -> tuple[float, np.ndarray]:
    """Return (overall MAE cm, per-measurement MAE cm) on a loader."""
    model.eval()
    abs_err = torch.zeros(len(MEAS_COLS))
    n = 0
    with torch.no_grad():
        for x, z in loader:
            pred = model(x.to(dev)).cpu()
            abs_err += (torch.abs(pred - z) * std_t).sum(0)
            n += x.shape[0]
    per = (abs_err / max(n, 1)).numpy()
    return float(per.mean()), per


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", type=Path, default=Path("data/bodym"))
    p.add_argument("--out", type=Path, default=Path("data/results/bmnet.pt"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=22)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-h", type=int, default=256)
    p.add_argument("--img-w", type=int, default=192,
                   help="per-view width; network input is 2*img_w wide")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--val-split", default="testA")
    args = p.parse_args(argv)

    dev = _device(args.device)
    print(f"device: {dev}")

    std = compute_standardizer(args.data_root, "train")
    print("measurement mean (cm):",
          np.round(std.meas_mean, 1).tolist())

    train_ds = BodyMDataset(args.data_root, "train", std,
                            img_h=args.img_h, img_w=args.img_w)
    val_ds = BodyMDataset(args.data_root, args.val_split, std,
                          img_h=args.img_h, img_w=args.img_w)
    print(f"train: {len(train_ds)} photos   "
          f"{args.val_split}: {len(val_ds)} photos")

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)

    model = BMnet(n_out=len(MEAS_COLS)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(0.75 * args.epochs), int(0.88 * args.epochs)],
        gamma=0.1)
    std_t = torch.from_numpy(std.meas_std.astype(np.float32))   # cm weights

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run = 0.0
        for x, z in train_ld:
            x, z = x.to(dev), z.to(dev)
            opt.zero_grad()
            pred = model(x)
            # L1 in centimetres: |z_pred - z| * meas_std.
            loss = (torch.abs(pred - z) * std_t.to(dev)).mean()
            loss.backward()
            opt.step()
            run += loss.detach().item() * x.shape[0]
        sched.step()
        train_mae = run / len(train_ds)
        val_mae, per = _evaluate(model, val_ld, std_t, dev)
        dt = time.time() - t0
        flag = ""
        if val_mae < best:
            best = val_mae
            torch.save({
                "state_dict": model.state_dict(),
                "standardizer": std.to_dict(),
                "img_h": args.img_h, "img_w": args.img_w,
                "meas_cols": list(MEAS_COLS),
                "val_mae_cm": val_mae,
            }, args.out)
            flag = "  *saved"
        print(f"ep {ep:3d}/{args.epochs}  train {train_mae:5.2f}  "
              f"val {val_mae:5.2f} cm  ({dt:4.0f}s){flag}")

    print(f"\nbest {args.val_split} MAE: {best:.2f} cm  ->  {args.out}")
    # Per-measurement breakdown from the best checkpoint.
    ck = torch.load(args.out, map_location=dev, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    _, per = _evaluate(model, val_ld, std_t, dev)
    print("per-measurement MAE (cm):")
    for name, e in zip(MEAS_COLS, per):
        print(f"  {name:<20} {e:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
