#!/usr/bin/env python3
"""
Train Phase 2 run-3 heuristic model (classifier or regressor) with:
  - 272-dim input: 256 one-hot + 16 per-cell Manhattan distances
  - Residual target: model predicts r(s) = h*(s) − MD_sum_excl_blank(s)

The residual is smaller (≈0–44 vs 0–80), reducing variance and allowing the
classifier to use 45 bins instead of 81.  At Phase 3 inference, restore h_hat:
    h_hat(s) = model.predict(x) + manhattan_sum(state)

Examples:
    python train_v2.py --config configs/classifier_v2.yaml
    python train_v2.py --config configs/regressor_v2.yaml --epochs 20
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).parent))

from data.puzzle_dataset import CostBalancedSampler, train_val_test_split
from data.puzzle_dataset_v2 import PuzzleDatasetV2, INPUT_DIM
from model import PinballLoss, PuzzleClassifierV2, PuzzleRegressorV2, NUM_RESIDUALS


# ── Helpers ────────────────────────────────────────────────────────────────────

class _SubsetDataset(Dataset):
    """Wraps a Subset to expose .costs for CostBalancedSampler."""

    def __init__(self, subset: Subset) -> None:
        self.costs   = subset.dataset.costs[np.array(subset.indices)]
        self._subset = subset

    def __len__(self) -> int:
        return len(self._subset)

    def __getitem__(self, idx: int):
        return self._subset[idx]


def _build_model(name: str) -> nn.Module:
    return PuzzleClassifierV2() if name == "classifier" else PuzzleRegressorV2()


def _build_loss(name: str, tau: float) -> nn.Module:
    return nn.CrossEntropyLoss() if name == "classifier" else PinballLoss(tau=tau)


def _residual_target(model_name: str, y: torch.Tensor, md_sum: torch.Tensor) -> torch.Tensor:
    """Compute loss target from raw cost and MD sum."""
    r = y - md_sum                                              # float residual
    if model_name == "classifier":
        return r.round().clamp(0, NUM_RESIDUALS - 1).long()    # integer class label
    return r                                                    # float for pinball


def _to_preds(
    model_name: str,
    out: torch.Tensor,
    cdf_threshold: "float | None" = None,
) -> torch.Tensor:
    """Convert raw model output to residual predictions (float [B])."""
    if model_name == "classifier":
        if cdf_threshold is not None:
            cdf = torch.softmax(out, dim=-1).cumsum(dim=-1)
            return (cdf < cdf_threshold).sum(dim=-1).clamp(max=NUM_RESIDUALS - 1).float()
        return out.argmax(dim=-1).float()
    # regressor: raw scalar, clamp residual to valid range
    return out.squeeze(-1).clamp(0, NUM_RESIDUALS - 1)


# ── Train / evaluate ───────────────────────────────────────────────────────────

_TRAIN_LOG_EVERY = 2_000
_EVAL_LOG_EVERY  =   500


def train_epoch(
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> float:
    model.train()
    total_loss, n = 0.0, 0
    n_batches = len(loader)
    t0 = time.time()

    for batch_idx, (x, y, md_sum) in enumerate(loader):
        x, y, md_sum = x.to(device), y.to(device), md_sum.to(device)
        optimizer.zero_grad()
        out    = model(x)
        target = _residual_target(model_name, y, md_sum)
        loss   = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
        n          += len(x)

        if (batch_idx + 1) % _TRAIN_LOG_EVERY == 0 or batch_idx == n_batches - 1:
            elapsed  = time.time() - t0
            sps      = n / elapsed
            eta_min  = (n_batches - batch_idx - 1) * (elapsed / (batch_idx + 1)) / 60
            pct      = (batch_idx + 1) / n_batches * 100
            print(
                f"  [train {pct:5.1f}%] batch {batch_idx+1}/{n_batches}"
                f"  loss={total_loss/n:.4f}"
                f"  {sps:,.0f} samp/s"
                f"  ETA {eta_min:.1f} min",
                flush=True,
            )

    scheduler.step()
    return total_loss / n


@torch.no_grad()
def evaluate(
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    prefix: str = "val",
    cdf_threshold: "float | None" = None,
) -> dict:
    """
    Evaluate the model against the true cost h*(s).

    Loss is computed against the residual target (model's actual objective).
    All other metrics (MAE, admissibility) are in the original h*(s) space:
      h_hat(s) = predicted_residual + md_sum
      err       = h_hat − h*   (positive = overestimate = inadmissible)
    """
    model.eval()
    total_loss       = 0.0
    sum_abs_err      = 0.0
    sum_sq_err       = 0.0
    sum_signed_err   = 0.0
    n_inadmissible   = 0
    sum_overestimate = 0.0
    max_overestimate = 0.0
    n                = 0
    n_batches        = len(loader)
    t0               = time.time()

    for batch_idx, (x, y, md_sum) in enumerate(loader):
        x, y, md_sum = x.to(device), y.to(device), md_sum.to(device)
        out    = model(x)

        # Loss against residual target
        target = _residual_target(model_name, y, md_sum)
        loss   = loss_fn(out, target)
        total_loss += loss.item() * len(x)

        # Admissibility metrics in h*(s) space
        pred_residual = _to_preds(model_name, out, cdf_threshold)
        h_hat         = pred_residual + md_sum   # reconstruct full heuristic estimate
        err           = h_hat - y                # positive = overestimate

        sum_abs_err    += err.abs().sum().item()
        sum_sq_err     += (err ** 2).sum().item()
        sum_signed_err += err.sum().item()

        over_mask = err > 0
        n_over    = int(over_mask.sum().item())
        n_inadmissible += n_over
        if n_over > 0:
            over_err          = err[over_mask]
            sum_overestimate += over_err.sum().item()
            max_overestimate  = max(max_overestimate, over_err.max().item())

        n += len(x)

        if (batch_idx + 1) % _EVAL_LOG_EVERY == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - t0
            pct     = (batch_idx + 1) / n_batches * 100
            print(
                f"  [{prefix} {pct:5.1f}%] batch {batch_idx+1}/{n_batches}"
                f"  mae={sum_abs_err/n:.3f}"
                f"  {n/elapsed:,.0f} samp/s",
                flush=True,
            )

    inadmissibility_rate = n_inadmissible / n
    return {
        f"{prefix}_loss":                total_loss / n,
        f"{prefix}_mae":                 sum_abs_err / n,
        f"{prefix}_rmse":                (sum_sq_err / n) ** 0.5,
        f"{prefix}_mean_signed_error":   sum_signed_err / n,
        f"{prefix}_admissibility_rate":  1.0 - inadmissibility_rate,
        f"{prefix}_inadmissibility_rate": inadmissibility_rate,
        f"{prefix}_overestimate_mean":   sum_overestimate / n_inadmissible if n_inadmissible > 0 else 0.0,
        f"{prefix}_overestimate_max":    max_overestimate,
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_history(history: list, train_losses: list, results_dir: Path) -> None:
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history) + 1))

    def _save(fig, name):
        fig.tight_layout()
        fig.savefig(plots_dir / name, dpi=120)
        plt.close(fig)
        print(f"  [plot] {plots_dir / name}")

    def _get(key):
        return [m[key] for m in history]

    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses,     label="train loss")
    ax.plot(epochs, _get("val_loss"), label="val loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Training vs Validation Loss (residual target)"); ax.legend(); ax.grid(True)
    _save(fig, "loss.png")

    fig, ax = plt.subplots()
    ax.plot(epochs, _get("val_mae"),  label="val MAE (h* space)")
    ax.plot(epochs, _get("val_rmse"), label="val RMSE (h* space)")
    ax.set_xlabel("epoch"); ax.set_ylabel("moves")
    ax.set_title("MAE and RMSE  (h_hat = residual + MD_sum)"); ax.legend(); ax.grid(True)
    _save(fig, "mae_rmse.png")

    fig, ax = plt.subplots()
    ax.plot(epochs, _get("val_mean_signed_error"), label="mean signed error")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="zero bias")
    ax.set_xlabel("epoch"); ax.set_ylabel("moves  (h_hat − h*)")
    ax.set_title("Prediction Bias  (positive = over-estimates)"); ax.legend(); ax.grid(True)
    _save(fig, "bias.png")

    fig, ax = plt.subplots()
    ax.plot(epochs, _get("val_admissibility_rate"),   label="admissible (%)")
    ax.plot(epochs, _get("val_inadmissibility_rate"), label="inadmissible (%)")
    ax.set_xlabel("epoch"); ax.set_ylabel("fraction"); ax.set_ylim(0, 1)
    ax.set_title("Admissibility Rate  (h_hat ≤ h*?)"); ax.legend(); ax.grid(True)
    _save(fig, "admissibility.png")

    fig, ax = plt.subplots()
    ax.plot(epochs, _get("val_overestimate_mean"), label="overestimate mean")
    ax.plot(epochs, _get("val_overestimate_max"),  label="overestimate max")
    ax.set_xlabel("epoch"); ax.set_ylabel("moves")
    ax.set_title("Overestimation Magnitude  (when inadmissible)"); ax.legend(); ax.grid(True)
    _save(fig, "overestimate.png")


# ── Checkpointing ──────────────────────────────────────────────────────────────

def save_checkpoint(path: Path, model: nn.Module, cfg: dict, epoch: int, val_mae: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_mae": val_mae},
        path,
    )
    print(f"  [ckpt] saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    yaml_defaults: dict = {}
    if pre_args.config:
        import yaml
        with open(pre_args.config) as f:
            yaml_defaults = yaml.safe_load(f) or {}

    p = argparse.ArgumentParser(description="Train Phase 2 run-3 heuristic model (v2)")
    p.add_argument("--config",          type=str,   default=None)
    p.add_argument("--model",           choices=["classifier", "regressor"])
    p.add_argument("--data",            type=str,   help=".bin file or directory")
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--batch-size",      type=int,   default=1024)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--tau",             type=float, default=0.3)
    p.add_argument("--out",             type=str,   default="checkpoints_v2")
    p.add_argument("--results",         type=str,   default="results/run3")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--num-workers",     type=int,   default=4)
    p.add_argument("--snapshot-every",  type=int,   default=5)
    p.add_argument("--cdf-threshold",   type=float, default=None)
    # V2-specific
    p.add_argument("--residual-target", action="store_true", default=True,
                   help="Train on r(s)=h*(s)−MD_sum instead of h*(s) directly (default: True)")
    p.add_argument("--no-residual-target", dest="residual_target", action="store_false")
    p.add_argument("--input-dim",       type=int,   default=INPUT_DIM,
                   help="Input dimension (default: 272)")

    p.set_defaults(**yaml_defaults)
    args = p.parse_args()

    if not args.model:
        p.error("--model is required (or set model: in the config file)")
    if not args.data:
        p.error("--data is required (or set data: in the config file)")

    return args


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print(
        f"device={device}  model={args.model}  epochs={args.epochs}"
        f"  residual_target={args.residual_target}  input_dim={args.input_dim}"
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading dataset …")
    ds                           = PuzzleDatasetV2(args.data, normalize_cost=False)
    train_sub, val_sub, test_sub = train_val_test_split(ds, seed=args.seed)
    train_ds                     = _SubsetDataset(train_sub)

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        sampler     = CostBalancedSampler(train_ds, num_samples=len(train_ds), seed=args.seed),
        num_workers = args.num_workers,
        pin_memory  = (device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_sub,
        batch_size  = args.batch_size * 2,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = (device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_sub,
        batch_size  = args.batch_size * 2,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = (device.type == "cuda"),
    )
    print(f"train={len(train_ds):,}  val={len(val_sub):,}  test={len(test_sub):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model    = _build_model(args.model).to(device)
    loss_fn  = _build_loss(args.model, args.tau)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    out_dir     = Path(args.out)     / args.model
    results_dir = Path(args.results) / args.model
    best_mae    = float("inf")
    cfg         = vars(args)
    history     = []
    train_losses = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(args.model, model, train_loader, loss_fn, optimizer, scheduler, device)
        metrics    = evaluate(args.model, model, val_loader, loss_fn, device, prefix="val",
                              cdf_threshold=args.cdf_threshold)
        val_mae    = metrics["val_mae"]

        train_losses.append(train_loss)
        history.append(metrics)

        print(
            f"epoch {epoch:3d}/{args.epochs}"
            f"  loss={train_loss:.4f}"
            f"  mae={metrics['val_mae']:.3f}"
            f"  rmse={metrics['val_rmse']:.3f}"
            f"  bias={metrics['val_mean_signed_error']:+.3f}"
            f"  inadmiss={metrics['val_inadmissibility_rate']:.3f}"
            f"  over_mean={metrics['val_overestimate_mean']:.2f}"
            f"  over_max={metrics['val_overestimate_max']:.0f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            save_checkpoint(out_dir / "best_model.pt", model, cfg, epoch, val_mae)

        if epoch % args.snapshot_every == 0:
            save_checkpoint(
                out_dir / f"epoch_{epoch:03d}_mae{val_mae:.3f}.pt", model, cfg, epoch, val_mae
            )

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_history(history, train_losses, results_dir)

    # ── Final test evaluation ─────────────────────────────────────────────────
    print(f"\n── Final test evaluation (best checkpoint, val MAE={best_mae:.3f}) ──")
    best_ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state"])
    test_metrics = evaluate(args.model, model, test_loader, loss_fn, device, prefix="test",
                            cdf_threshold=args.cdf_threshold)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    import json
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"  [metrics] saved → {metrics_path}")


if __name__ == "__main__":
    main()
