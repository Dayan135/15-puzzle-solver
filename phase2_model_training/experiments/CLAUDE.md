# Phase 2 — Post-Training Experiments

This directory documents standalone evaluation experiments that run on already-trained checkpoints
**without retraining**. Each experiment is a single Python script in `phase2_model_training/`.

---

## Experiment: Classifier Threshold × Temperature Sweep

**Script:** `phase2_model_training/evaluate_thresholds.py`

**Purpose:**  
Map the full admissibility / MAE tradeoff curve for `PuzzleClassifier` by sweeping two
inference-time parameters:

| Parameter | Role | Default range |
|---|---|---|
| `cdf_threshold` | Quantile of the predicted cost distribution to return as h(s) | 0.05 – 0.50, step 0.05 |
| `temperature` | Divides logits before softmax; T > 1 flattens overconfident distributions | 1.0, 1.5, 2.0, 3.0 |

Neither parameter requires retraining. The saved `best_model.pt` is loaded once;
all logits are collected in a single forward pass and reused across the grid.

**Key background:**

- `PuzzleClassifier` outputs logits over 81 cost classes (0–80).
- At inference, `predict_quantile(threshold, temperature)` returns the smallest class `k`
  such that `P(cost ≤ k | temperature) ≥ threshold`.
- A lower threshold predicts a lower quantile → more admissible but higher MAE (under-estimates more).
- Temperature T > 1 spreads probability mass, reducing single-bin overconfidence.
  This lowers `overestimate_max` without changing the model weights.
- `threshold ≠ admissibility rate` — CrossEntropyLoss produces uncalibrated probabilities.
  The sweep finds the *actual* threshold that achieves 95% or 99% admissibility on the data.

**Run command (local):**
```bash
cd phase2_model_training
python evaluate_thresholds.py \
    --checkpoint checkpoints/classifier/best_model.pt \
    --data /path/to/dataset_000.bin \
    --split test                     # or val
```

**Run command (cluster — `jobs/evaluate_thresholds.sh`):**
```bash
cd phase2_model_training

# Default: test split, full threshold/temperature grid, best_model.pt
sbatch jobs/evaluate_thresholds.sh

# Val split first — select threshold without touching test numbers
SPLIT=val sbatch jobs/evaluate_thresholds.sh

# Fine-grained sweep around the 99% admissibility region (once val run reveals the area)
THRESHOLDS="0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10" \
sbatch jobs/evaluate_thresholds.sh

# Different checkpoint (epoch snapshot) → separate output dir to avoid overwriting
CHECKPOINT=checkpoints/classifier/epoch_010_mae1.013.pt \
OUT_DIR=results/classifier_epoch010 \
sbatch jobs/evaluate_thresholds.sh
```

SLURM resources: RTX 3090, 32 GB RAM (needed for 100M-record dataset load), 8 CPUs,
2-hour walltime. The forward pass over 4.2M test samples takes under 5 minutes; the
remaining time covers dataset loading (~10–15 s) and the grid evaluation.

**Outputs** (under `results/classifier/`):
```
threshold_sweep.json                         metrics for every (threshold, temp) pair
plots/threshold_sweep_admissibility.png      admissibility vs threshold, curve per temp
plots/threshold_sweep_mae.png                MAE vs threshold, curve per temp
plots/threshold_sweep_frontier.png           admissibility vs MAE scatter (Pareto frontier)
plots/threshold_sweep_over_max.png           worst-case overestimate vs threshold
```

**How to modify this experiment:**

| Goal | Change |
|---|---|
| Finer threshold grid | `--thresholds 0.01 0.02 0.03 ...` |
| More temperatures | `--temperatures 1.0 1.2 1.5 2.0 2.5 3.0` |
| Evaluate on val (for hyperparameter selection) | `--split val` |
| Evaluate a different checkpoint (e.g. epoch 10 snapshot) | `--checkpoint checkpoints/classifier/epoch_010_mae1.013.pt` |
| Different output directory | `--out-dir results/classifier_run3` |
| Faster run (larger batch) | `--batch-size 4096` |

**Interpreting results for Phase 3:**

1. Find the operating points that reach ≥95% and ≥99% admissibility.
2. Record the corresponding `(threshold, temperature, MAE)` triple.
3. Set those values in the Phase 3 search config so the classifier heuristic operates
   at a known, measured admissibility level.
4. Also keep the threshold=0.5, T=1.0 point as a "high-accuracy" baseline.

**Example table output:**
```
thresh  temp  admiss%  inadmiss%    MAE   RMSE    bias  over_mean  over_max
  0.05   1.0   99.XX%     0.XX%  X.XXX  X.XXX  -X.XXX     X.XXXX      X.0
  0.10   1.0   98.XX%     1.XX%  X.XXX  ...
  ...
  0.50   1.0   76.69%    23.31%  0.996  1.544  +0.041     2.2252     50.0
```

**Known results (run 2 checkpoint, threshold=0.50, T=1.0, test split):**
```
admissibility_rate:   76.69%
MAE:                   0.996
overestimate_max:     50.0 moves
```
These are the baseline numbers. The sweep fills in the rest of the curve.
