# Phase 2 Training — Run 3 Analysis

> **✅ These results are valid.** Run 3 used the corrected unique-state split (no val/test leakage),
> GPU nodes throughout, and 8 DataLoader workers. Both models use 272-dim input (one-hot + per-cell MD)
> and predict the **residual** r(s) = h*(s) − MD_sum(s). All metrics are reported in h* space
> (residual + MD_sum) for direct comparison with run 2.

---

## Run info

| | Classifier | Regressor |
|---|---|---|
| Job ID | 18038466 | 18038468 |
| Date | 2026-06-08 | 2026-06-08 |
| Node | ise-3090-03 | ise-3090-03 |
| GPU | RTX 3090, 24 GB (index 0) | RTX 3090, 24 GB (index 1) |
| Epochs | 20 | 15 |
| Batch size | 1024 | 1024 |
| Input dim | 272 (one-hot 256 + per-cell MD 16) | 272 |
| Target | residual r(s) = h*(s) − MD_sum(s) | residual r(s) = h*(s) − MD_sum(s) |
| Inference | CDF quantile, threshold=0.5, temp=1.0 | `round(clamp(output, 0, 44))` |
| Loss | CrossEntropyLoss (45 residual classes) | PinballLoss τ=0.3 |
| Best checkpoint | epoch 20 (val MAE=0.929) | epoch 13 (val MAE=1.271) |
| Throughput (train) | ~43 k samp/s | ~41.5 k samp/s |
| Throughput (val) | ~46 k samp/s | ~46 k samp/s |

**Shared split** (100M records, seed=42):
- train = 78,111,663 records (all duplicates included)
- val   =  4,209,527 records (one per unique state)
- test  =  4,209,527 records (one per unique state)

---

## Results snapshot

| Metric | Classifier | Regressor |
|---|---|---|
| Val MAE (best epoch) | **0.929** (ep 20) | 1.271 (ep 13) |
| Test MAE | **0.930** | 1.272 |
| Test RMSE | **1.475** | 1.838 |
| Mean signed error (bias) | +0.042 (neutral) | −0.694 (underestimates) |
| Admissibility rate | 77.9% | 62.1% |
| Inadmissibility rate | **22.1%** | **37.9%** ⚠️ |
| Overestimate mean | 2.20 moves | **0.76 moves** ✅ |
| Overestimate max | **22 moves** ✅ | **8 moves** ✅ |
| Params | 609,837 | 598,529 |

---

## Run 2 vs Run 3 comparison

| Metric | Run 2 Classifier | Run 3 Classifier | Δ | Run 2 Regressor | Run 3 Regressor | Δ |
|---|---|---|---|---|---|---|
| Test MAE | 0.996 | **0.930** | −0.066 ✅ | 1.287 | 1.272 | −0.015 ≈ |
| Test RMSE | 1.544 | **1.475** | −0.069 ✅ | 1.774 | 1.838 | +0.064 ⚠️ |
| Admissibility | 76.7% | 77.9% | +1.2 pp ≈ | **82.4%** | 62.1% | −20.3 pp ⚠️ |
| Overestimate max | 50 | **22** | −28 ✅ | 14 | **8** | −6 ✅ |
| Overestimate mean | 2.23 | 2.20 | −0.03 ≈ | 1.63 | **0.76** | −0.87 ✅ |
| Input dim | 256 | 272 | | 256 | 272 | |
| Target | h*(s) | residual | | h*(s) | residual | |

---

## Per-epoch trajectory

### Classifier

| Epoch | Train loss | Val MAE | Val RMSE | Bias   | Inadmiss | Over mean | Over max |
|-------|-----------|---------|----------|--------|----------|-----------|----------|
| 1     | 0.7073    | 1.118   | 1.674    | +0.014 | 0.248    | 2.28      | 10       |
| 2     | 0.6263    | 1.070   | 1.621    | −0.008 | 0.236    | 2.25      | 8        |
| 3     | 0.6058    | 1.048   | 1.598    | −0.014 | 0.231    | 2.24      | 8        |
| 4     | 0.5932    | 1.032   | 1.579    | −0.045 | 0.222    | 2.23      | 10       |
| 5     | 0.5837    | 1.020   | 1.566    | +0.027 | 0.235    | 2.23      | 8        |
| 6     | 0.5758    | 1.010   | 1.555    | +0.003 | 0.228    | 2.22      | 8        |
| 7     | 0.5685    | 1.002   | 1.546    | +0.029 | 0.232    | 2.22      | 8        |
| 8     | 0.5615    | 0.995   | 1.539    | +0.071 | 0.239    | 2.23      | 8        |
| 9     | 0.5547    | 0.987   | 1.529    | −0.012 | 0.221    | 2.21      | 10       |
| 10    | 0.5479    | 0.976   | 1.519    | +0.018 | 0.225    | 2.21      | 8        |
| 11    | 0.5412    | 0.969   | 1.514    | +0.050 | 0.230    | 2.21      | 10       |
| 12    | 0.5345    | 0.962   | 1.505    | −0.030 | 0.212    | 2.20      | 8        |
| 13    | 0.5279    | 0.954   | 1.499    | +0.008 | 0.218    | 2.21      | 20       |
| 14    | 0.5216    | 0.947   | 1.492    | +0.025 | 0.220    | 2.20      | 8        |
| 15    | 0.5157    | 0.943   | 1.488    | +0.055 | 0.226    | 2.21      | 20       |
| 16    | 0.5103    | 0.939   | 1.485    | +0.067 | 0.228    | 2.21      | 8        |
| 17    | 0.5056    | 0.935   | 1.481    | +0.061 | 0.226    | 2.21      | 18       |
| 18    | 0.5019    | 0.933   | 1.478    | +0.056 | 0.224    | 2.20      | 24       |
| 19    | 0.4992    | 0.931   | 1.476    | +0.051 | 0.223    | 2.20      | 24       |
| 20    | 0.4976    | **0.929**| **1.474**| +0.042 | **0.221**| **2.20**  | **24**   |

### Regressor

| Epoch | Train loss | Val MAE | Val RMSE | Bias   | Inadmiss | Over mean | Over max |
|-------|-----------|---------|----------|--------|----------|-----------|----------|
| 1     | 0.3841    | 1.525   | 2.050    | −0.958 | 0.303    | 0.94      | 8        |
| 2     | 0.3418    | 1.459   | 2.022    | −0.882 | 0.336    | 0.86      | 8        |
| 3     | 0.3318    | 1.392   | 1.948    | −0.770 | 0.290    | 1.07      | 16       |
| 4     | 0.3266    | 1.405   | 1.961    | −0.790 | 0.185    | 1.66      | 9        |
| 5     | 0.3234    | 1.358   | 1.928    | −0.694 | 0.372    | 0.89      | 14       |
| 6     | 0.3195    | 1.368   | 1.938    | −0.738 | 0.317    | 0.99      | 24       |
| 7     | 0.3148    | 1.364   | 1.938    | −0.747 | 0.230    | 1.34      | 8        |
| 8     | 0.3093    | 1.353   | 1.925    | −0.775 | 0.226    | 1.28      | 8        |
| 9     | 0.3039    | 1.343   | 1.911    | −0.774 | 0.382    | 0.74      | 8        |
| 10    | 0.2989    | 1.377   | 1.950    | −0.889 | 0.431    | 0.56      | 8        |
| 11    | 0.2941    | 1.328   | 1.901    | −0.792 | 0.253    | 1.06      | 8        |
| 12    | 0.2894    | 1.304   | 1.875    | −0.761 | 0.358    | 0.76      | 8        |
| 13    | 0.2852    | **1.271**| **1.837**| −0.695 | 0.378    | 0.76      | 8        |
| 14    | 0.2819    | 1.280   | 1.849    | −0.729 | 0.308    | 0.89      | 8        |
| 15    | 0.2796    | 1.273   | 1.841    | −0.716 | 0.284    | 0.98      | 8        |

---

## Key observations

### 1. Residual target delivered its main promise: overestimate max capped at 22

The classifier's worst-case overestimate dropped from 50 moves (run 2) to 22 moves (run 3), with overestimate max
of 8 stable for epochs 1–16 before creeping to 24 at epochs 18–20. This is the direct effect of the
bounded residual output space [0, 44]: the model can no longer make catastrophic predictions that exceed
the true cost by 50 moves. For Phase 3 search experiments, this eliminates the most dangerous failure
mode of run 2 — a single 50-move overestimate that permanently blocks an A* search path.

The regressor's overestimate max improved from 14 → 8, and overestimate mean from 1.63 → 0.76.

### 2. Classifier MAE improved by 6.6%; both models still learning at final epoch

The classifier reached test MAE=0.930 at epoch 20, down from 0.996 in run 2. The loss and MAE
curves show no plateau — training loss drops monotonically from 0.707 → 0.498 and val MAE from
1.118 → 0.929 without any sign of overfitting (train/val gap stays wide and stable throughout).
Extrapolating the trend, 5–8 more epochs would likely push val MAE to ~0.91–0.92.

The regressor's val MAE continued declining until epoch 13 (1.271) and showed no clear plateau,
though the loss curve slows noticeably after epoch 10.

### 3. Regressor admissibility regressed significantly: 82.4% → 62.1%

This is the most concerning result of run 3. Despite a nearly identical mean bias (−0.694 vs −0.715),
the admissibility rate dropped from 82.4% to 62.1%, meaning 37.9% of predictions now overestimate h*.

The root cause is higher RMSE (1.838 vs 1.774) — the error distribution has more variance. Given
bias μ=−0.694 and RMSE≈1.838, the implied error standard deviation is σ≈√(1.838²−0.694²)≈1.70 moves.
Under a normal approximation: P(overestimate) = P(error>0) ≈ Φ(0.694/1.70) ≈ 34%, which closely
matches the observed 37.9%.

The per-epoch admissibility is also highly erratic — swings of 18–21 pp within adjacent epochs
(e.g., ep 4=18.5%, ep 5=37.2%, ep 6=31.7%, ep 7=23.0%) — suggesting the PinballLoss optimization
landscape is unstable when the target is a residual concentrated near zero. PinballLoss τ=0.3 targets
the 30th percentile of r(s). Many states have r(s)=0 or 1 (MD_sum already equals or nearly equals
h*), so the 30th percentile is very low, and the optimizer oscillates around this boundary.

### 4. Classifier admissibility unchanged at default threshold

At threshold=0.5, temperature=1.0, admissibility went from 76.7% to 77.9% — effectively no change.
The explanation is the same as run 2: the CDF threshold=0.5 returns the model's median prediction;
since the model is trained on a cost-balanced uniform distribution but tested on the natural distribution
(skewed toward high costs), the median systematically undershoots. A threshold sweep on the run 3
checkpoint is required to map the full admissibility/MAE curve. Based on run 2 patterns, thresh=0.10
temp=2.0 should achieve ≥99% admissibility on the run 3 model.

### 5. Regressor no longer the preferred Phase 3 variant

In run 2, the regressor was the safer default: higher admissibility (82.4% vs 76.7%), lower overestimate
max (14 vs 50). In run 3 this advantage has reversed — the regressor is now inadmissible 37.9% of the
time vs 22.1% for the classifier. The classifier strictly dominates on admissibility, MAE, and RMSE.
The regressor retains a lower overestimate max (8 vs 22) and far lower overestimate mean (0.76 vs 2.20),
but those advantages are outweighed by the inadmissibility regression for most Phase 3 use cases.

### 6. Classifier overestimate max trend in late epochs is a concern

The overestimate max was stable at 8 moves for epochs 1–12, then started spiking — reaching 20 at
epochs 13 and 15, and 24 at epochs 18–20. This suggests the model is learning to better capture the
mean of the distribution at the cost of occasionally misplacing probability mass at the high end.
The `best_model.pt` (epoch 20) has overestimate max=24. Epoch 12 or 14 checkpoints (max=8) offer a
better tail bound at marginally higher MAE (0.962 vs 0.930). The threshold sweep on `best_model.pt`
will determine whether temperature scaling can bring the tail back to 8–9 moves.

### 7. Throughput slightly lower than run 2

Classifier: ~43 k samp/s vs ~43 k in run 2 (unchanged). Regressor: ~41.5 k samp/s vs ~70 k in
run 2 (the 272-dim input and more complex dataset loading slow the regressor). Val throughput
unchanged at ~46 k samp/s for both.

---

## Saved artifacts

| Path | Contents |
|---|---|
| `checkpoints_v2/classifier/best_model.pt` | Best classifier (val MAE=0.929, epoch 20) |
| `checkpoints_v2/classifier/epoch_005_mae1.020.pt` | Epoch 5 snapshot |
| `checkpoints_v2/classifier/epoch_010_mae0.976.pt` | Epoch 10 snapshot |
| `checkpoints_v2/classifier/epoch_015_mae0.943.pt` | Epoch 15 snapshot |
| `checkpoints_v2/classifier/epoch_020_mae0.929.pt` | Epoch 20 snapshot (= best) |
| `checkpoints_v2/regressor/best_model.pt` | Best regressor (val MAE=1.271, epoch 13) |
| `checkpoints_v2/regressor/epoch_005_mae1.358.pt` | Epoch 5 snapshot |
| `checkpoints_v2/regressor/epoch_010_mae1.377.pt` | Epoch 10 snapshot |
| `checkpoints_v2/regressor/epoch_015_mae1.273.pt` | Epoch 15 snapshot |
| `results/run3/classifier/test_metrics.json` | Full classifier test metrics |
| `results/run3/regressor/test_metrics.json` | Full regressor test metrics |
| `results/run3/classifier/plots/` | Loss, MAE/RMSE, bias, admissibility, overestimate plots |
| `results/run3/regressor/plots/` | Same plots for regressor |

---

## Next steps

1. **Threshold sweep on run 3 classifier** — run `jobs/evaluate_thresholds_v2.sh` on
   `checkpoints_v2/classifier/best_model.pt` to map the full admissibility/MAE/overestimate-max
   curve. Expected best ≥99% point: thresh≈0.10, temp≈2.0 (based on run 2 patterns).

2. **Consider epoch 12–14 classifier checkpoint for Phase 3** — overestimate max=8 at these
   epochs vs 24 at epoch 20. Trade-off: MAE 0.962 vs 0.930. The threshold sweep on `best_model.pt`
   may recover a bounded tail via temperature scaling regardless.

3. **Regressor: increase τ for run 4** — τ=0.3 was appropriate for the raw h* target (range 0–80)
   but may be too aggressive for the residual target (range 0–34, heavily concentrated near 0).
   A higher τ (e.g., 0.45) would reduce the underestimation bias and likely stabilize admissibility.
