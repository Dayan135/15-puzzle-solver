# Phase 2 Training — Run 2 Analysis

> **✅ These results are valid.** Run 2 used the corrected unique-state split (no val/test leakage),
> GPU nodes throughout, and 8 DataLoader workers. Both models are meaningful baselines for Phase 3.

---

## Run info

| | Classifier | Regressor |
|---|---|---|
| Job ID | 18018359 | 18018360 |
| Date | 2026-06-06 → 06-07 | 2026-06-06 |
| Node | ise-3090-03 | dt-3090-01 |
| GPU | RTX 3090, 24 GB | RTX 3090, 24 GB |
| Epochs | 14 | 10 |
| Batch size | 1024 | 1024 |
| Inference | CDF quantile, threshold=0.5 | `round(clamp(output, 0, 80))` |
| Loss | CrossEntropyLoss | PinballLoss τ=0.3 |

**Shared split** (100M records, seed=42):
- train = 78,111,663 records (all duplicates included)
- val   =  4,209,527 records (one per unique state)
- test  =  4,209,527 records (one per unique state)

---

## Results snapshot

| Metric | Classifier | Regressor |
|---|---|---|
| Val MAE (best epoch) | **0.995** (ep 14) | 1.286 (ep 10) |
| Test MAE | **0.996** | 1.287 |
| Test RMSE | **1.544** | 1.774 |
| Mean signed error (bias) | +0.041 (neutral) | −0.715 (underestimates) |
| Admissibility rate | 76.7% | **82.4%** |
| Inadmissibility rate | **23.3%** | 17.6% |
| Overestimate mean | 2.23 moves | **1.63 moves** |
| Overestimate max | **50 moves** ⚠️ | **14 moves** ✅ |
| Throughput (val) | ~84 k samp/s | ~190 k samp/s |
| Params | 614,993 | 594,433 |

---

## Per-epoch trajectory

### Classifier

| Epoch | Train loss | Val MAE | Val RMSE | Bias   | Inadmiss | Over mean | Over max |
|-------|-----------|---------|----------|--------|----------|-----------|----------|
| 1     | 0.7361    | 1.174   | 1.728    | +0.170 | 0.291    | 2.31      | 10       |
| 2     | 0.6493    | 1.115   | 1.666    | +0.011 | 0.249    | 2.26      | 42       |
| 3     | 0.6258    | 1.095   | 1.648    | +0.073 | 0.257    | 2.27      | 52       |
| 4     | 0.6107    | 1.076   | 1.625    | +0.111 | 0.262    | 2.27      | 22       |
| 5     | 0.5993    | 1.061   | 1.611    | +0.031 | 0.242    | 2.25      | 44       |
| 6     | 0.5896    | 1.049   | 1.598    | −0.018 | 0.230    | 2.24      | 42       |
| 7     | 0.5811    | 1.038   | 1.585    | +0.000 | 0.232    | 2.24      | 40       |
| 8     | 0.5732    | 1.028   | 1.572    | +0.006 | 0.232    | 2.23      | 24       |
| 9     | 0.5659    | 1.020   | 1.563    | +0.009 | 0.231    | 2.22      | 42       |
| 10    | 0.5592    | 1.013   | 1.557    | +0.012 | 0.230    | 2.22      | 30       |
| 11    | 0.5531    | 1.007   | 1.553    | +0.062 | 0.240    | 2.23      | 70       |
| 12    | 0.5477    | 1.002   | 1.551    | +0.069 | 0.240    | 2.23      | 70       |
| 13    | 0.5435    | 0.998   | 1.546    | +0.044 | 0.234    | 2.23      | 58       |
| 14    | 0.5409    | **0.995**| **1.544**| +0.040 | 0.233    | 2.22      | 58       |

### Regressor

| Epoch | Train loss | Val MAE | Val RMSE | Bias   | Inadmiss | Over mean | Over max |
|-------|-----------|---------|----------|--------|----------|-----------|----------|
| 1     | 0.4123    | 1.441   | 1.906    | −0.791 | 0.213    | 1.52      | 9        |
| 2     | 0.3443    | 1.452   | 1.925    | −0.905 | 0.180    | 1.52      | 8        |
| 3     | 0.3302    | 1.357   | 1.814    | −0.735 | 0.202    | 1.54      | 8        |
| 4     | 0.3219    | 1.364   | 1.825    | −0.790 | 0.187    | 1.54      | 16       |
| 5     | 0.3127    | 1.360   | 1.836    | −0.825 | 0.171    | 1.56      | 16       |
| 6     | 0.3041    | 1.318   | 1.798    | −0.746 | 0.180    | 1.59      | 16       |
| 7     | 0.2983    | 1.324   | 1.805    | −0.781 | 0.171    | 1.59      | 16       |
| 8     | 0.2936    | 1.317   | 1.801    | −0.782 | 0.168    | 1.59      | 19       |
| 9     | 0.2896    | 1.289   | 1.774    | −0.721 | 0.176    | 1.62      | 16       |
| 10    | 0.2870    | **1.286**| **1.773**| −0.716 | **0.175**| **1.63**  | **16**   |

---

## Key observations

### 1. CDF inference is active — and correctly so

`cdf_threshold=0.5` was set in `configs/classifier.yaml` and propagated through `args.cdf_threshold`
into every `evaluate()` call. All classifier metrics in this log reflect the CDF quantile (median)
inference, not argmax. This is confirmed by the dramatically lower bias vs run 1 (argmax was used
there and produced +7 move bias with a 42-move overestimate max).

### 2. Admissibility at threshold=0.5 is 76.7% — this is expected, not a failure

Threshold=0.5 predicts the *model's median*. If the model were perfectly calibrated, its median
would exceed the true cost exactly 50% of the time (i.e., 50% inadmissible). Seeing only 23.3%
inadmissible means the model's probability mass is systematically shifted low — the model is
conservative. This is a known consequence of CostBalancedSampler training on a uniform cost
distribution while the test set reflects the natural (higher-cost) distribution.

For Phase 3 experiments, threshold=0.5 gives one operating point on the admissibility/accuracy
curve. Lower thresholds will push toward ≥99% admissibility. **A threshold sweep on the saved
checkpoint (no retraining needed) will map the full curve** — see `evaluate_thresholds.py`.

### 3. Overestimate max of 50 is the alarming number

Even at the median, some states get predictions 50 moves above the true cost. For WA*/GBFS,
a single state catastrophically overestimated will never be expanded, potentially blocking
solution paths. The regressor's max is capped at 14 by PinballLoss's structural asymmetry.
The classifier needs a lower threshold (or temperature scaling) to contain the tail.

### 4. Regressor wins on admissibility and bounded overestimation

Despite higher MAE (1.287 vs 0.996), the regressor:
- Is admissible 82.4% of the time vs 76.7%
- Never overestimates by more than 14 moves (vs 50 for the classifier)
- Has mean overestimate of only 1.63 moves vs 2.23

This aligns with the PinballLoss design: τ=0.3 penalizes overestimates 2.3× more than
underestimates, structurally biasing the model toward the 30th percentile.

For WA* admissibility guarantees, the regressor is the safer choice at the current operating
point. The classifier at a lower CDF threshold could match or exceed it.

### 5. Both models are still learning — more epochs needed

**Classifier:** MAE drops monotonically 1.174 → 0.995 across 14 epochs with no plateau.
Extrapolating the trend suggests ~4–6 more epochs would give MAE ≈ 0.97–0.98.

**Regressor:** More erratic (1.441 → 1.286), with a small plateau around epochs 6–8 before
dropping again at epoch 9. The Pinball loss landscape is rougher than CrossEntropy. At least
5 more epochs are warranted to check for further improvement.

### 6. Regressor is 2× faster at inference

At 190 k samp/s vs 84 k samp/s on the same GPU. CrossEntropyLoss requires a 81-class softmax
for both training and val metrics; PinballLoss uses raw scalar output. This advantage matters
in Phase 3 when the heuristic is called for every expanded node.

---

## Run 1 vs Run 2 comparison (classifier only — run 1 regressor had same data bug)

| Metric | Run 1 (contaminated) | Run 2 (valid) | Note |
|---|---|---|---|
| Test MAE | 0.431 | 0.996 | Run 1 was optimistically biased by leakage |
| Admissibility | 90.1% | 76.7% | Run 1 threshold was not set (argmax used) |
| Overestimate max | 42 | 50 | CDF inference helps but doesn't eliminate tail |
| Epochs | 10 | 14 | Both show no plateau |

The large MAE drop (0.431 → 0.996) confirms run 1 was severely contaminated by leakage.

---

## Saved artifacts

| Path | Contents |
|---|---|
| `checkpoints/classifier/best_model.pt` | Best classifier (val MAE=0.995, epoch 14) |
| `checkpoints/classifier/epoch_005_mae1.061.pt` | Epoch 5 snapshot |
| `checkpoints/classifier/epoch_010_mae1.013.pt` | Epoch 10 snapshot |
| `checkpoints/regressor/best_model.pt` | Best regressor (val MAE=1.286, epoch 10) |
| `checkpoints/regressor/epoch_005_mae1.360.pt` | Epoch 5 snapshot |
| `checkpoints/regressor/epoch_010_mae1.286.pt` | Epoch 10 snapshot |
| `results/classifier/test_metrics.json` | Full classifier test metrics |
| `results/regressor/test_metrics.json` | Full regressor test metrics |
| `results/classifier/plots/` | Loss, MAE/RMSE, bias, admissibility, overestimate plots |
| `results/regressor/plots/` | Same plots for regressor |
