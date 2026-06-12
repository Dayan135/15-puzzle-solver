# Phase 2 Training — Run 4 Classifier Analysis

> **Run 4 change:** MD per-cell distances normalised by dividing by 6 (`/6`) before concatenation
> with the one-hot features. All other settings identical to run 3 (20 epochs, τ irrelevant for
> classifier, CDF threshold=0.5 for training metrics). Threshold sweep on the run 4 checkpoint
> completed post-training (job 18122447).

---

## Run info

| | Classifier |
|---|---|
| Job ID | 18106638 |
| Threshold sweep job | 18122447 |
| Date | 2026-06-11 |
| Node | cs-3090-01 |
| GPU | RTX 3090, 24 GB (index 3) |
| Epochs | 20 |
| Batch size | 1024 |
| Input dim | 272 (one-hot 256 + per-cell MD 16, normalised /6) |
| Target | residual r(s) = h*(s) − MD_sum(s) |
| Inference | CDF quantile, threshold=0.5, temp=1.0 (training default) |
| Loss | CrossEntropyLoss (45 residual classes) |
| Best checkpoint | epoch 20 (val MAE=0.933) |

**Shared split** (100M records, seed=42):
- train = 78,111,663 records (all duplicates included)
- val   =  4,209,527 records (one per unique state)
- test  =  4,209,527 records (one per unique state)

---

## Test results

| Metric | Run 3 | Run 4 | Δ |
|---|---|---|---|
| Test MAE | 0.9298 | **0.9348** | +0.005 ≈ |
| Test RMSE | 1.4754 | **1.4812** | +0.006 ≈ |
| Mean signed error (bias) | +0.042 | **+0.037** | −0.005 ≈ |
| Admissibility rate | 77.93% | **77.97%** | +0.04 pp ≈ |
| Inadmissibility rate | 22.07% | **22.02%** | −0.05 pp ≈ |
| Overestimate mean | 2.203 | **2.205** | ≈ |
| **Overestimate max** | **22** | **16** | **−6 ✅** |

**Main result:** MAE, RMSE, and admissibility are essentially unchanged — MD normalisation does not
affect accuracy at the default threshold. The meaningful win is `over_max` dropping from 22 to 16,
continuing the trend from run 2 (50) → run 3 (22) → run 4 (16).

---

## Per-epoch trajectory

| Epoch | Train loss | Val MAE | Val RMSE | Bias   | Inadmiss | Over mean | Over max |
|-------|-----------|---------|----------|--------|----------|-----------|----------|
| 1     | 0.7000    | 1.120   | 1.676    | +0.098 | 0.266    | 2.29      | 8        |
| 2     | 0.6230    | 1.072   | 1.624    | +0.092 | 0.256    | 2.27      | 8        |
| 3     | 0.6030    | 1.048   | 1.596    | +0.054 | 0.245    | 2.25      | 8        |
| 4     | 0.5913    | 1.031   | 1.575    | −0.000 | 0.231    | 2.23      | 12       |
| 5     | 0.5826    | 1.019   | 1.564    | +0.010 | 0.231    | 2.23      | 12       |
| 6     | 0.5753    | 1.010   | 1.554    | −0.033 | 0.221    | 2.22      | 8        |
| 7     | 0.5685    | 1.004   | 1.551    | −0.015 | 0.223    | 2.22      | 8        |
| 8     | 0.5618    | 0.995   | 1.541    | −0.035 | 0.217    | 2.21      | 8        |
| 9     | 0.5554    | 0.987   | 1.533    | −0.062 | 0.210    | 2.21      | 8        |
| 10    | 0.5489    | 0.980   | 1.525    | −0.015 | 0.218    | 2.21      | 10       |
| 11    | 0.5424    | 0.972   | 1.518    | +0.004 | 0.221    | 2.21      | 8        |
| 12    | 0.5359    | 0.965   | 1.512    | +0.016 | 0.222    | 2.21      | 8        |
| 13    | 0.5295    | 0.959   | 1.506    | +0.054 | 0.229    | 2.21      | 8        |
| 14    | 0.5234    | 0.954   | 1.500    | +0.072 | 0.232    | 2.21      | **22**   |
| 15    | 0.5175    | 0.948   | 1.494    | +0.068 | 0.230    | 2.21      | 18       |
| 16    | 0.5121    | 0.943   | 1.490    | +0.058 | 0.227    | 2.21      | 18       |
| 17    | 0.5074    | 0.940   | 1.486    | +0.062 | 0.227    | 2.21      | 22       |
| 18    | 0.5036    | 0.937   | 1.483    | +0.062 | 0.226    | 2.21      | 18       |
| 19    | 0.5009    | 0.934   | 1.480    | +0.045 | 0.222    | 2.20      | 18       |
| 20    | 0.4993    | **0.933** | **1.479** | +0.035 | **0.220** | **2.20** | **18** |

Compared to run 3, the `over_max` trajectory is better throughout: epochs 1–13 stay at ≤12 (vs ≤20
in run 3), and late-epoch spikes reach 22 at epochs 14 and 17 (vs 24 at epochs 18–20 in run 3).
The test `over_max=16` reflects this improvement.

---

## Threshold sweep (job 18122447, test split, seed=42)

Full grid: thresholds 0.05–0.50 (step 0.05) × temperatures {1.0, 1.5, 2.0, 3.0}.

```
----------------------------------------------------------------------------------
  thresh   temp   admiss%  inadmiss%     MAE    RMSE    bias  over_mean  over_max
----------------------------------------------------------------------------------
    0.05    1.0    98.42%      1.58%  1.8745  2.4600 -1.8092     2.0681      16.0
    0.10    1.0    96.77%      3.23%  1.5366  2.1185 -1.4024     2.0753      16.0
    0.15    1.0    94.96%      5.04%  1.3371  1.9126 -1.1265     2.0879      16.0
    0.20    1.0    93.00%      7.00%  1.2017  1.7697 -0.9076     2.1007      16.0
    0.25    1.0    90.89%      9.11%  1.1059  1.6666 -0.7206     2.1156      16.0
    0.30    1.0    88.65%     11.35%  1.0364  1.5906 -0.5527     2.1311      16.0
    0.35    1.0    86.24%     13.76%  0.9873  1.5367 -0.3965     2.1476      16.0
    0.40    1.0    83.67%     16.33%  0.9552  1.5014 -0.2483     2.1649      16.0
    0.45    1.0    80.94%     19.06%  0.9378  1.4832 -0.1050     2.1845      16.0
    0.50    1.0    77.98%     22.02%  0.9348  1.4812 +0.0366     2.2055      16.0
----------------------------------------------------------------------------------
    0.05    1.5    99.39%      0.61%  2.3251  2.9062 -2.2999     2.0796      16.0
    0.10    1.5    98.40%      1.60%  1.8617  2.4439 -1.7955     2.0642      16.0
    0.15    1.5    97.04%      2.96%  1.5739  2.1535 -1.4513     2.0691      16.0
    0.20    1.5    95.34%      4.66%  1.3690  1.9428 -1.1751     2.0810      16.0
    0.25    1.5    93.31%      6.69%  1.2177  1.7835 -0.9374     2.0938      16.0
    0.30    1.5    90.94%      9.06%  1.1066  1.6639 -0.7245     2.1094      16.0
    0.35    1.5    88.28%     11.72%  1.0261  1.5755 -0.5275     2.1269      16.0
    0.40    1.5    85.26%     14.74%  0.9719  1.5156 -0.3393     2.1463      16.0
    0.45    1.5    81.91%     18.09%  0.9414  1.4823 -0.1567     2.1686      16.0
    0.50    1.5    78.20%     21.80%  0.9331  1.4743 +0.0232     2.1937      16.0
----------------------------------------------------------------------------------
    0.05    2.0    99.72%      0.28%  2.7182  3.2979 -2.7065     2.1149      16.0
    0.10    2.0    99.14%      0.86%  2.1527  2.7333 -2.1172     2.0654      16.0
    0.15    2.0    98.15%      1.85%  1.7929  2.3735 -1.7168     2.0601      16.0
    0.20    2.0    96.73%      3.27%  1.5299  2.1068 -1.3949     2.0667      16.0
    0.25    2.0    94.87%      5.13%  1.3292  1.8993 -1.1157     2.0789      16.0
    0.30    2.0    92.52%      7.48%  1.1780  1.7386 -0.8649     2.0928      16.0
    0.35    2.0    89.69%     10.31%  1.0676  1.6182 -0.6325     2.1106      16.0
    0.40    2.0    86.39%     13.61%  0.9921  1.5342 -0.4117     2.1317      16.0
    0.45    2.0    82.56%     17.44%  0.9482  1.4853 -0.1965     2.1556      16.0
    0.50    2.0    78.23%     21.77%  0.9355  1.4719 +0.0155     2.1839      16.0
----------------------------------------------------------------------------------
    0.05    3.0    99.91%      0.09%  3.4152  4.0118 -3.4112     2.2163      12.0
    0.10    3.0    99.70%      0.30%  2.6743  3.2577 -2.6618     2.0932      14.0
    0.15    3.0    99.19%      0.81%  2.1916  2.7747 -2.1584     2.0561      16.0
    0.20    3.0    98.24%      1.76%  1.8296  2.4120 -1.7575     2.0495      16.0
    0.25    3.0    96.72%      3.28%  1.5453  2.1225 -1.4105     2.0565      16.0
    0.30    3.0    94.53%      5.47%  1.3224  1.8896 -1.0959     2.0687      16.0
    0.35    3.0    91.57%      8.43%  1.1545  1.7083 -0.8028     2.0858      16.0
    0.40    3.0    87.82%     12.18%  1.0384  1.5790 -0.5251     2.1079      16.0
    0.45    3.0    83.26%     16.74%  0.9702  1.5013 -0.2551     2.1359      16.0
    0.50    3.0    77.84%     22.16%  0.9500  1.4778 +0.0121     2.1705      16.0
----------------------------------------------------------------------------------

Operating points closest to admissibility targets:
  target    thresh   temp   actual%     MAE  over_max
     90%      0.25    1.0    90.89%  1.1059      16.0
     95%      0.20    1.5    95.34%  1.3690      16.0
     99%      0.10    2.0    99.14%  2.1527      16.0
```

### Key finding: over_max=16 is a hard ceiling

`over_max=16` across all 40 operating points except T=3.0/thresh=0.05 (12) and T=3.0/thresh=0.10 (14).
This is a global property of the model, not a threshold artefact: the residual output space is
bounded at [0, 44] and the model has learned to avoid placing significant probability mass above 16
on test states. Temperature scaling shifts probability mass toward the uniform distribution, reducing
the tail risk slightly at very high T, but the ceiling is already low enough to be acceptable for
Phase 3 search.

### Phase 3 operating point recommendations

| Use case | thresh | T | Admissibility | MAE | over_max | Rationale |
|---|---|---|---|---|---|---|
| **Near-optimal WA*** | 0.20 | 1.5 | 95.34% | 1.369 | 16 | ≥95% admissible; w=1.1 near-optimal |
| **Balanced WA*** | 0.25 | 1.0 | 90.89% | 1.106 | 16 | Best MAE at ≥90%; fast, decent quality |
| **Aggressive GBFS** | 0.50 | 1.0 | 77.98% | 0.935 | 16 | Lowest MAE; admissibility irrelevant for GBFS |
| **Nearly-admissible** | 0.05 | 3.0 | 99.91% | 3.415 | 12 | Closest to true A*; use for baseline |

The **balanced WA* point (thresh=0.25, T=1.0)** is likely the best single starting point for Phase 3:
it achieves the lowest MAE among the ≥90% admissible operating points and requires no temperature
rescaling. The near-optimal point (thresh=0.20, T=1.5) adds ~5 pp admissibility at the cost of
+0.26 MAE — worth testing for experiments that require tight suboptimality bounds.

---

## Key observations

### 1. MD normalisation tightened the worst-case overestimate

`over_max` dropped from 22 (run 3 test) to 16 (run 4 test) despite identical MAE and admissibility.
The trajectory shows the same pattern: run 3 saw max spikes of 24 at late epochs; run 4 spikes reach
22 at epochs 14 and 17 but settle at 18 for the best checkpoint. The mechanism is that dividing MD
distances by 6 reduces the dynamic range of the MD feature component, which constrained the model's
ability to place probability mass at extreme residual values.

### 2. MAE and admissibility unchanged at default threshold

At thresh=0.5, T=1.0, run 4 MAE (0.9348) and admissibility (77.97%) are within noise of run 3
(0.9298, 77.93%). The normalisation only affected the model's output distribution tail, not its
central tendency. Both models are still learning at epoch 20 — the train loss curve shows no plateau
(0.700 → 0.499) and val MAE continues declining monotonically.

### 3. Threshold sweep confirms over_max=16 is a model-level property

The threshold sweep on the run 4 checkpoint shows `over_max=16` is uniform across all operating
points (except the extreme T=3.0 corners). This is in contrast to run 3, where `over_max=22` in the
default configuration and temperature scaling was proposed as a potential remedy. Run 4 resolves this
without requiring non-default temperatures: the default (thresh=0.5, T=1.0) already yields max=16.

### 4. Model still improving at epoch 20

The loss curve from 0.700 → 0.499 and val MAE from 1.120 → 0.933 show no sign of convergence.
Extrapolating the trend, 5–10 more epochs would likely push val MAE below 0.92. However, the
overestimate-max tail starts appearing at epoch 14 — running beyond 20 epochs risks pushing
`over_max` higher, as seen in run 3. The current checkpoint is a reasonable stopping point.

---

## Saved artifacts

| Path | Contents |
|---|---|
| `checkpoints_v2/classifier/best_model.pt` | Best classifier (val MAE=0.933, epoch 20) |
| `checkpoints_v2/classifier/epoch_005_mae1.019.pt` | Epoch 5 snapshot |
| `checkpoints_v2/classifier/epoch_010_mae0.980.pt` | Epoch 10 snapshot |
| `checkpoints_v2/classifier/epoch_015_mae0.948.pt` | Epoch 15 snapshot |
| `checkpoints_v2/classifier/epoch_020_mae0.933.pt` | Epoch 20 snapshot (= best) |
| `results/run4/classifier/test_metrics.json` | Full classifier test metrics |
| `results/run4/classifier/threshold_sweep.json` | Full threshold × temperature sweep results |
| `results/run4/classifier/plots/` | Loss, MAE/RMSE, bias, admissibility, overestimate plots |
| `results/run4/classifier/plots/threshold_sweep_*.png` | Threshold sweep visualisations |
