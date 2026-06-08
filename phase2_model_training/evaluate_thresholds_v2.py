#!/usr/bin/env python3
"""
Sweep CDF threshold × softmax temperature for a trained PuzzleClassifierV2.

V2 difference from evaluate_thresholds.py:
  - Uses PuzzleDatasetV2 (272-dim input)
  - Model outputs residual logits; h_hat = predicted_residual + md_sum
  - Admissibility is evaluated in h*(s) space, not residual space

Usage:
    cd phase2_model_training
    python evaluate_thresholds_v2.py \\
        --checkpoint checkpoints_v2/classifier/best_model.pt \\
        --data /path/to/dataset_000.bin

    # Val split for threshold selection
    SPLIT=val python evaluate_thresholds_v2.py \\
        --checkpoint checkpoints_v2/classifier/best_model.pt \\
        --data /path/to/dataset_000.bin \\
        --split val

Outputs (all relative to --out-dir, default results/run3/classifier):
    threshold_sweep.json
    plots/threshold_sweep_admissibility.png
    plots/threshold_sweep_mae.png
    plots/threshold_sweep_frontier.png
    plots/threshold_sweep_over_max.png

SLURM:
    cd phase2_model_training
    sbatch jobs/evaluate_thresholds_v2.sh
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data.puzzle_dataset import train_val_test_split
from data.puzzle_dataset_v2 import PuzzleDatasetV2
from model import PuzzleClassifierV2, NUM_RESIDUALS

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_THRESHOLDS   = [round(t, 2) for t in np.arange(0.05, 0.55, 0.05).tolist()]
DEFAULT_TEMPERATURES = [1.0, 1.5, 2.0, 3.0]
BATCH_SIZE           = 2048


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_logits_targets_mdsums(
    model: PuzzleClassifierV2,
    loader: DataLoader,
    device: torch.device,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Single forward pass; returns (logits [N,45], targets [N], md_sums [N])."""
    model.eval()
    logits_list, targets_list, md_list = [], [], []
    n_batches = len(loader)
    for i, (x, y, md_sum) in enumerate(loader):
        logits_list.append(model(x.to(device)).cpu())
        targets_list.append(y.cpu())
        md_list.append(md_sum.cpu())
        if (i + 1) % 200 == 0 or i == n_batches - 1:
            pct = (i + 1) / n_batches * 100
            print(f"  [forward {pct:5.1f}%] batch {i+1}/{n_batches}", flush=True)
    return (
        torch.cat(logits_list,  dim=0),
        torch.cat(targets_list, dim=0),
        torch.cat(md_list,      dim=0),
    )


def evaluate_grid(
    logits:       torch.Tensor,   # [N, 45]
    targets:      torch.Tensor,   # [N]  — true h*(s)
    md_sums:      torch.Tensor,   # [N]  — MD_sum_excl_blank
    thresholds:   list[float],
    temperatures: list[float],
) -> dict:
    """Evaluate all (threshold, temperature) pairs.

    preds = residual CDF quantile; h_hat = preds + md_sums.
    All metrics are in h*(s) space so results are directly comparable to v1.
    """
    results = {}
    for temp in temperatures:
        scaled  = logits / temp
        softmax = torch.softmax(scaled, dim=-1)          # [N, 45]
        cdf     = softmax.cumsum(dim=-1)                 # [N, 45]

        for thresh in thresholds:
            pred_residual = (cdf < thresh).sum(dim=-1).clamp(max=NUM_RESIDUALS - 1).float()
            h_hat         = pred_residual + md_sums.float()   # reconstruct full estimate
            err           = h_hat - targets.float()            # positive = overestimate

            over_mask = err > 0
            n_over    = int(over_mask.sum().item())
            n         = len(targets)

            results[(thresh, temp)] = {
                "threshold":            thresh,
                "temperature":          temp,
                "mae":                  err.abs().mean().item(),
                "rmse":                 (err ** 2).mean().sqrt().item(),
                "bias":                 err.mean().item(),
                "admissibility_rate":   1.0 - n_over / n,
                "inadmissibility_rate": n_over / n,
                "overestimate_mean":    err[over_mask].mean().item() if n_over > 0 else 0.0,
                "overestimate_max":     err[over_mask].max().item()  if n_over > 0 else 0.0,
            }
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_table(results: dict, thresholds: list, temperatures: list) -> None:
    header = (
        f"{'thresh':>8} {'temp':>6} {'admiss%':>9} {'inadmiss%':>10}"
        f" {'MAE':>7} {'RMSE':>7} {'bias':>7} {'over_mean':>10} {'over_max':>9}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for temp in temperatures:
        for thresh in thresholds:
            m = results[(thresh, temp)]
            print(
                f"{thresh:>8.2f} {temp:>6.1f}"
                f" {m['admissibility_rate']*100:>8.2f}%"
                f" {m['inadmissibility_rate']*100:>9.2f}%"
                f" {m['mae']:>7.4f}"
                f" {m['rmse']:>7.4f}"
                f" {m['bias']:>+7.4f}"
                f" {m['overestimate_mean']:>10.4f}"
                f" {m['overestimate_max']:>9.1f}"
            )
        print(sep)


def find_admissibility_targets(results: dict, targets: list[float]) -> None:
    print("\nOperating points closest to admissibility targets:")
    print(f"  {'target':>8}  {'thresh':>8} {'temp':>6} {'actual%':>9} {'MAE':>7} {'over_max':>9}")
    for target in targets:
        best = min(
            results.values(),
            key=lambda m: abs(m["admissibility_rate"] - target)
                          if m["admissibility_rate"] >= target
                          else float("inf"),
        )
        if best["admissibility_rate"] < target:
            print(f"  {target*100:>7.0f}%   (no operating point reaches this level)")
        else:
            print(
                f"  {target*100:>7.0f}%"
                f"  {best['threshold']:>8.2f}"
                f" {best['temperature']:>6.1f}"
                f" {best['admissibility_rate']*100:>8.2f}%"
                f" {best['mae']:>7.4f}"
                f" {best['overestimate_max']:>9.1f}"
            )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(
    results:      dict,
    thresholds:   list[float],
    temperatures: list[float],
    out_dir:      Path,
) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _save(fig, name):
        fig.tight_layout()
        fig.savefig(plots_dir / name, dpi=120)
        plt.close(fig)
        print(f"  [plot] {plots_dir / name}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for temp in temperatures:
        admiss = [results[(t, temp)]["admissibility_rate"] * 100 for t in thresholds]
        ax.plot(thresholds, admiss, marker="o", label=f"T={temp}")
    ax.axhline(99, color="gray", linestyle="--", linewidth=0.8, label="99% target")
    ax.axhline(95, color="silver", linestyle=":", linewidth=0.8, label="95% target")
    ax.set_xlabel("CDF threshold"); ax.set_ylabel("Admissibility (%)")
    ax.set_title("ClassifierV2: Admissibility vs CDF Threshold × Temperature")
    ax.legend(); ax.grid(True); ax.set_ylim(0, 102)
    _save(fig, "threshold_sweep_admissibility.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for temp in temperatures:
        maes = [results[(t, temp)]["mae"] for t in thresholds]
        ax.plot(thresholds, maes, marker="o", label=f"T={temp}")
    ax.set_xlabel("CDF threshold"); ax.set_ylabel("MAE (moves, h* space)")
    ax.set_title("ClassifierV2: MAE vs CDF Threshold × Temperature")
    ax.legend(); ax.grid(True)
    _save(fig, "threshold_sweep_mae.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D"]
    for i, temp in enumerate(temperatures):
        admiss = [results[(t, temp)]["admissibility_rate"] * 100 for t in thresholds]
        maes   = [results[(t, temp)]["mae"] for t in thresholds]
        ax.scatter(maes, admiss, marker=markers[i % len(markers)], label=f"T={temp}", s=60)
        for j, thresh in enumerate(thresholds):
            ax.annotate(f"{thresh:.2f}", (maes[j], admiss[j]),
                        textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.axhline(99, color="gray", linestyle="--", linewidth=0.8, label="99% target")
    ax.set_xlabel("MAE (moves, lower = better)"); ax.set_ylabel("Admissibility (%)")
    ax.set_title("ClassifierV2: Admissibility vs MAE Tradeoff (labels = threshold)")
    ax.legend(); ax.grid(True)
    _save(fig, "threshold_sweep_frontier.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for temp in temperatures:
        over_max = [results[(t, temp)]["overestimate_max"] for t in thresholds]
        ax.plot(thresholds, over_max, marker="o", label=f"T={temp}")
    ax.set_xlabel("CDF threshold"); ax.set_ylabel("Overestimate max (moves)")
    ax.set_title("ClassifierV2: Worst-Case Overestimate vs Threshold × Temperature")
    ax.legend(); ax.grid(True)
    _save(fig, "threshold_sweep_over_max.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ClassifierV2 threshold × temperature sweep")
    p.add_argument("--checkpoint",    required=True,  help="Path to best_model.pt")
    p.add_argument("--data",          required=True,  help=".bin file or directory")
    p.add_argument("--thresholds",    nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    p.add_argument("--temperatures",  nargs="+", type=float, default=DEFAULT_TEMPERATURES)
    p.add_argument("--split",         choices=["val", "test"], default="test")
    p.add_argument("--batch-size",    type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers",   type=int, default=4)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--out-dir",       type=str, default="results/run3/classifier")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  split={args.split}")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg       = ckpt.get("cfg", {})
    width     = cfg.get("width", 256)
    depth     = cfg.get("depth", 4)
    input_dim = cfg.get("input_dim", 272)
    model     = PuzzleClassifierV2(width=width, depth=depth, input_dim=input_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(
        f"  model: width={width} depth={depth} input_dim={input_dim}"
        f"  trained to epoch={ckpt.get('epoch')}  val_mae={ckpt.get('val_mae', '?'):.4f}"
    )

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading dataset …")
    ds = PuzzleDatasetV2(args.data, normalize_cost=False)
    _, val_sub, test_sub = train_val_test_split(ds, seed=args.seed)
    split_ds = val_sub if args.split == "val" else test_sub
    print(f"  {args.split} split: {len(split_ds):,} records")

    loader = DataLoader(
        split_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = (device.type == "cuda"),
    )

    # ── Forward pass (once for all combos) ───────────────────────────────────
    print("Running forward pass …")
    logits, targets, md_sums = collect_logits_targets_mdsums(model, loader, device)
    print(f"  logits: {logits.shape}  targets: {targets.shape}  md_sums: {md_sums.shape}")

    # ── Grid evaluation ──────────────────────────────────────────────────────
    print(f"\nEvaluating {len(args.thresholds)} thresholds × {len(args.temperatures)} temperatures …")
    results = evaluate_grid(logits, targets, md_sums, args.thresholds, args.temperatures)

    # ── Print table ──────────────────────────────────────────────────────────
    print()
    print_table(results, args.thresholds, args.temperatures)
    find_admissibility_targets(results, [0.90, 0.95, 0.99])

    # ── Save JSON ────────────────────────────────────────────────────────────
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "threshold_sweep.json"
    serializable = {
        f"thresh={k[0]:.2f}_temp={k[1]:.1f}": v for k, v in results.items()
    }
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  [json] saved → {json_path}")

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_results(results, args.thresholds, args.temperatures, out_dir)


if __name__ == "__main__":
    main()
