"""Train BMnet on the BodyM dataset, with optional ABS augmentation.

Follows the paper's recipe (§3.1, §5): L1 loss on the 14 measurements,
Adam at 1e-3, batch 22, a multi-step learning-rate schedule dropping
×0.1 at 75 % and 88 % of training, and best-model selection on a held-out
10 % of the training set. The reporting split (BodyM TestA by default)
is kept untouched for the final per-measurement table.

With ``--abs`` the adversarial body simulator (``bmnet.abs``) is enabled,
reproducing the paper's three-phase schedule (§3.2):

  1. pre-train on real BodyM photos (``--epochs``);
  2. fine-tune for ``--abs-epochs`` epochs on adversarial synthetic
     bodies — each batch is rendered from SMPL-X shapes driven by
     gradient ascent on the *current* BMnet's loss, freshly sampled
     every epoch (never repeated);
  3. a short real-data fine-tune to bridge the synthetic→real domain
     gap (``--abs-real-epochs``).

The loss is evaluated in centimetres: the network emits standardized
measurements, ``|z_pred - z_true| * meas_std`` recovers the cm error,
and that is what is back-propagated and reported.

  python -m tailor_twin.bmnet.train --epochs 80 --out data/results/bmnet.pt
  python -m tailor_twin.bmnet.train --epochs 80 --abs --out data/results/bmnet.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from . import MEAS_COLS
from .dataset import BodyMDataset, compute_standardizer
from .model import BMnet

# Fraction of the training set held out for best-model selection (§5).
HOLDOUT_FRAC = 0.10


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


def _train_pass(model, opt, batches, std_t, dev) -> float:
    """One epoch over an iterable of ``(x, z)`` batches. The BMnet
    training loss is the L1 measurement error in centimetres
    (``|z_pred - z| * meas_std``). Returns the mean cm error."""
    model.train()
    sw = std_t.to(dev)
    run, n = 0.0, 0
    for x, z in batches:
        x, z = x.to(dev), z.to(dev)
        opt.zero_grad()
        pred = model(x)
        loss = (torch.abs(pred - z) * sw).mean()
        loss.backward()
        opt.step()
        run += loss.detach().item() * x.shape[0]
        n += x.shape[0]
    return run / max(n, 1)


def _save(path, model, std, args, val_mae) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "standardizer": std.to_dict(),
        "img_h": args.img_h, "img_w": args.img_w,
        "meas_cols": list(MEAS_COLS),
        "val_mae_cm": val_mae,
    }, path)


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
    p.add_argument("--report-split", default="testA",
                   help="held-out split for the final per-measure table")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-folder", default="data/body_models")
    # --- ABS adversarial augmentation -----------------------------------
    p.add_argument("--abs", action="store_true",
                   help="enable adversarial body simulator fine-tuning")
    p.add_argument("--abs-epochs", type=int, default=10,
                   help="phase 2: epochs of synthetic adversarial bodies")
    p.add_argument("--abs-batches", type=int, default=280,
                   help="adversarial batches per ABS epoch; the paper's "
                        "~10x-real-data regime is len(train)/batch_size")
    p.add_argument("--abs-real-epochs", type=int, default=5,
                   help="phase 3: real-data fine-tune epochs")
    p.add_argument("--abs-lr", type=float, default=1e-4,
                   help="learning rate for the ABS fine-tune phases")
    p.add_argument("--abs-num-betas", type=int, default=16,
                   help="SMPL-X shape betas the simulator varies")
    p.add_argument("--abs-seed-sigma", type=float, default=1.0,
                   help="std of the random seed shapes before ascent")
    p.add_argument("--abs-gender", default="neutral")
    args = p.parse_args(argv)

    dev = _device(args.device)
    print(f"device: {dev}")

    std = compute_standardizer(args.data_root, "train")
    print("measurement mean (cm):", np.round(std.meas_mean, 1).tolist())

    # Best-model selection uses a 10 % holdout of the training set; the
    # report split (TestA) stays untouched for the final table (§5).
    full_train = BodyMDataset(args.data_root, "train", std,
                              img_h=args.img_h, img_w=args.img_w)
    n_hold = max(1, int(HOLDOUT_FRAC * len(full_train)))
    n_tr = len(full_train) - n_hold
    g = torch.Generator().manual_seed(args.seed)
    train_ds, hold_ds = random_split(full_train, [n_tr, n_hold],
                                     generator=g)
    report_ds = BodyMDataset(args.data_root, args.report_split, std,
                             img_h=args.img_h, img_w=args.img_w)
    print(f"train: {n_tr}   holdout(val): {n_hold}   "
          f"{args.report_split}: {len(report_ds)}")

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    hold_ld = DataLoader(hold_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers)
    report_ld = DataLoader(report_ds, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.workers)

    model = BMnet(n_out=len(MEAS_COLS)).to(dev)
    std_t = torch.from_numpy(std.meas_std.astype(np.float32))   # cm weights

    # ------------------------------------------------------------------
    # phase 1 — supervised pre-training on real BodyM photos
    # ------------------------------------------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(0.75 * args.epochs), int(0.88 * args.epochs)],
        gamma=0.1)
    best = float("inf")
    print("\n=== phase 1: supervised pre-training ===")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_mae = _train_pass(model, opt, train_ld, std_t, dev)
        sched.step()
        val_mae, _ = _evaluate(model, hold_ld, std_t, dev)
        flag = ""
        if val_mae < best:
            best = val_mae
            _save(args.out, model, std, args, val_mae)
            flag = "  *saved"
        print(f"ep {ep:3d}/{args.epochs}  train {train_mae:5.2f}  "
              f"val {val_mae:5.2f} cm  ({time.time() - t0:4.0f}s){flag}")

    # ------------------------------------------------------------------
    # phases 2 & 3 — adversarial body simulator (paper §3.2)
    # ------------------------------------------------------------------
    if args.abs:
        from .abs import AbsSampler

        # Resume from the best pre-trained weights.
        ck = torch.load(args.out, map_location=dev, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        sampler = AbsSampler(
            model_folder=args.model_folder, gender=args.abs_gender,
            num_betas=args.abs_num_betas, img_h=args.img_h,
            img_w=args.img_w, device=dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.abs_lr)

        print("\n=== phase 2: adversarial synthetic fine-tuning ===")
        for ep in range(1, args.abs_epochs + 1):
            t0 = time.time()
            # Fresh adversarial bodies every epoch — never repeated.
            batches = [sampler.make_batch(model, std, args.batch_size,
                                          sigma=args.abs_seed_sigma)
                       for _ in range(args.abs_batches)]
            train_mae = _train_pass(model, opt, batches, std_t, dev)
            val_mae, _ = _evaluate(model, hold_ld, std_t, dev)
            flag = ""
            if val_mae < best:
                best = val_mae
                _save(args.out, model, std, args, val_mae)
                flag = "  *saved"
            print(f"abs {ep:3d}/{args.abs_epochs}  synth {train_mae:5.2f}  "
                  f"val {val_mae:5.2f} cm  ({time.time() - t0:4.0f}s){flag}")

        print("\n=== phase 3: real-data fine-tuning ===")
        for ep in range(1, args.abs_real_epochs + 1):
            t0 = time.time()
            train_mae = _train_pass(model, opt, train_ld, std_t, dev)
            val_mae, _ = _evaluate(model, hold_ld, std_t, dev)
            flag = ""
            if val_mae < best:
                best = val_mae
                _save(args.out, model, std, args, val_mae)
                flag = "  *saved"
            print(f"real {ep:3d}/{args.abs_real_epochs}  "
                  f"train {train_mae:5.2f}  val {val_mae:5.2f} cm  "
                  f"({time.time() - t0:4.0f}s){flag}")

    # ------------------------------------------------------------------
    print(f"\nbest holdout MAE: {best:.2f} cm  ->  {args.out}")
    ck = torch.load(args.out, map_location=dev, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    report_mae, per = _evaluate(model, report_ld, std_t, dev)
    print(f"{args.report_split} MAE: {report_mae:.2f} cm")
    print("per-measurement MAE (cm):")
    for name, e in zip(MEAS_COLS, per):
        print(f"  {name:<20} {e:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
