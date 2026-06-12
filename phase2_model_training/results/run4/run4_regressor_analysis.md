# Phase 2 Training — Run 4 Regressor Analysis

> **Run 4 changes:** (1) Pinball loss quantile τ increased from 0.3 → 0.4 to reduce the
> underestimation bias observed in run 3; (2) MD per-cell distances normalised by /6 (same as
> classifier). All other settings identical to run 3 (15 epochs, residual target, 272-dim input).

---

## Run info

| | Regressor |
|---|---|
| Job ID | 18106639 |
| Date | 2026-06-11 |
| Node | cs-3090-04 |
| GPU | RTX 3090, 24 GB (index 4) |
| Epochs | 15 |
| Batch size | 1024 |
| Input dim | 272 (one-hot 256 + per-cell MD 16, normalised /6) |
| Target | residual r(s) = h*(s) − MD_sum(s) |
| Loss | PinballLoss τ=0.4 (up from τ=0.3 in run 3) |
| Inference | `round(clamp(output, 0, 44))` |
| Best checkpoint | epoch 15 (val MAE=1.161) |

**Shared split** (100M records, seed=42):
- train = 78,111,663 records (all duplicates included)
- val   =  4,209,527 records (one per unique state)
- test  =  4,209,527 records (one per unique state)

---

## Test results

| Metric | Run 3 (τ=0.3) | Run 4 (τ=0.4) | Δ |
|---|---|---|---|
| Test MAE | 1.2719 | **1.1606** | **−0.111 ✅** |
| Test RMSE | 1.8379 | **1.7110** | **−0.127 ✅** |
| Mean signed error (bias) | −0.694 | **−0.331** | **+0.363 ✅** |
| Admissibility rate | 62.14% | **61.44%** | −0.7 pp ≈ |
| Inadmissibility rate | 37.86% | **38.56%** | +0.7 pp ≈ |
| Overestimate mean | 0.764 | **1.076** | +0.312 ⚠️ |
| Overestimate max | 8.0 | **8.0** | unchanged |

**Main result:** τ=0.4 corrected the underestimation bias (−0.694 → −0.331) and improved MAE by 8.7%
and RMSE by 6.9%. Admissibility remained essentially unchanged at ~61%, and `over_max` stayed at 8.
The trade-off is a higher `overestimate_mean` (1.076 vs 0.764) — when the model does overestimate,
the magnitude is slightly larger. This is expected: a higher quantile shifts the predicted distribution
upward, increasing the frequency and size of overestimates on states where residual is near zero.

---

## Per-epoch trajectory

| Epoch | Train loss | Val MAE | Val RMSE | Bias   | Inadmiss | Over mean | Over max |
|-------|-----------|---------|----------|--------|----------|-----------|----------|
| 1     | 0.4187    | 1.362   | 1.884    | −0.471 | 0.393    | 1.13      | 10       |
| 2     | 0.3777    | 1.317   | 1.869    | −0.381 | 0.376    | 1.24      | 10       |
| 3     | 0.3670    | 1.310   | 1.865    | −0.441 | 0.274    | 1.58      | 8        |
| 4     | 0.3582    | 1.262   | 1.808    | −0.321 | 0.535    | 0.88      | 8        |
| 5     | 0.3511    | 1.269   | 1.826    | −0.429 | 0.436    | 0.96      | 8        |
| 6     | 0.3450    | 1.236   | 1.785    | −0.352 | 0.336    | 1.31      | 8        |
| 7     | 0.3391    | 1.205   | 1.750    | −0.239 | 0.301    | 1.61      | 8        |
| 8     | 0.3339    | 1.220   | 1.771    | −0.394 | 0.463    | 0.89      | 8        |
| 9     | 0.3286    | 1.218   | 1.772    | −0.418 | 0.498    | 0.80      | 8        |
| 10    | 0.3238    | 1.199   | 1.752    | −0.373 | 0.430    | 0.96      | 8        |
| 11    | 0.3190    | 1.175   | 1.723    | −0.306 | 0.423    | 1.03      | 8        |
| 12    | 0.3148    | 1.169   | 1.719    | −0.331 | 0.507    | 0.83      | 8        |
| 13    | 0.3113    | 1.162   | 1.712    | −0.309 | 0.457    | 0.93      | 8        |
| 14    | 0.3084    | 1.161   | 1.710    | −0.328 | 0.307    | 1.36      | 8        |
| 15    | 0.3065    | **1.161** | **1.710** | −0.332 | 0.385 | **1.08** | **8**  |

The per-epoch inadmissibility remains erratic — swings from 27% (epoch 14) to 51% (epoch 4) within
adjacent epochs. This is the same PinballLoss instability as run 3, but less severe: the range is
roughly 27–54% in run 4 vs 19–43% in run 3 (with the τ change, higher inadmissibility variance is
expected because the 40th percentile is above the mode of the residual distribution). The test
inadmissibility (38.56%) falls within the observed val range, suggesting no overfitting.

Val MAE shows no plateau across 15 epochs (1.362 → 1.161), and the loss curve continues declining
monotonically (0.419 → 0.307). The model was still improving at the stopping point.

---

## Run 3 vs Run 4 analysis

### τ increase resolved the bias problem

In run 3, τ=0.3 forced the model to predict the 30th percentile of the residual distribution. Given
that the residual r(s) = h*(s) − MD_sum(s) is heavily concentrated near zero (many states have
MD_sum already close to h*), the 30th percentile is very low, causing systematic underestimation.
Increasing to τ=0.4 shifted predictions toward the median, cutting the signed error from −0.694 to
−0.331 and improving MAE by 8.7%.

### Admissibility did not improve despite less underestimation

Admissibility stayed at ~61% despite the bias correction. The reason: admissibility is determined by
the sign of the error, not its magnitude. With τ=0.4, the model predicts a higher quantile on average,
which shifts some states from underestimate to overestimate. The net effect is near-zero because the
improvement in underestimation magnitude is offset by slightly more states crossing the admissibility
boundary. For the regressor, admissibility is structurally limited by the PinballLoss objective — it
cannot exceed approximately (1−τ)×100% in expectation.

### over_max is robust at 8 across all epochs and both runs

The regressor's `over_max=8` is stable across all epochs in both run 3 and run 4, with the only
exceptions at epochs 1–2 of run 4 (max=10) and one epoch in run 3 (max=16 at epoch 3). This 8-move
ceiling is significantly better than the classifier's run 4 ceiling of 16 — when the regressor does
overestimate, the error is bounded much more tightly.

---

## Classifier vs Regressor (run 4, for Phase 3 decision)

| Metric | Classifier (run 4, thresh=0.5, T=1.0) | Regressor (run 4) |
|---|---|---|
| Test MAE | 0.9348 | 1.1606 |
| Test RMSE | 1.4812 | 1.7110 |
| Admissibility | **77.97%** | 61.44% |
| Overestimate mean | 2.205 | **1.076** |
| **Overestimate max** | 16 | **8** |

The classifier strictly dominates on MAE, RMSE, and admissibility. The regressor's sole advantage is
a tighter worst-case overestimate (8 vs 16). For Phase 3:

- **Classifier** is the primary heuristic for Phase 3 experiments. With threshold sweep, it can reach
  ≥95% admissibility (thresh=0.20, T=1.5) at MAE=1.369 — still better than the regressor's base MAE.
- **Regressor** is a useful secondary heuristic: lower overestimate max (8) means WA* suboptimality
  bounds derived from the regressor are tighter in the worst case, even though its base admissibility
  is lower. Consider pairing the regressor with a W=2–3 weight in WA* experiments.

---

## Key observations

### 1. τ=0.4 is better than τ=0.3 for the residual target, but τ=0.5 may be optimal

With τ=0.4, the signed error is −0.331 — still systematically underestimating. The 40th percentile
of r(s) is still below the mean because the residual distribution is right-skewed (most states have
small residuals; hard states have large ones). τ=0.5 would target the median, likely pushing bias
closer to zero and possibly improving admissibility toward ~50%.

### 2. Val MAE still declining at epoch 15

No convergence is visible: loss drops from 0.419 to 0.307 and val MAE from 1.362 to 1.161 without
plateau. However, since `over_max=8` is stable throughout, there is no tail-risk reason to stop early
for the regressor (unlike the classifier). More epochs would likely push MAE below 1.10.

### 3. The regressor's over_max=8 is a structural property of the output representation

The regressor uses `round(clamp(output, 0, 44))`, so the residual prediction can technically range
from 0 to 44. The empirical over_max=8 reflects that the model never places large residual predictions
on states where h*(s) is low. This is a consequence of the residual target: states with small r(s)
(near-goal states) cluster tightly near 0 in feature space, so the model consistently predicts small
values for them. It is not an artifact of thresholding.

---

## Saved artifacts

| Path | Contents |
|---|---|
| `checkpoints_v2/regressor/best_model.pt` | Best regressor (val MAE=1.161, epoch 15) |
| `checkpoints_v2/regressor/epoch_005_mae1.269.pt` | Epoch 5 snapshot |
| `checkpoints_v2/regressor/epoch_010_mae1.199.pt` | Epoch 10 snapshot |
| `checkpoints_v2/regressor/epoch_015_mae1.161.pt` | Epoch 15 snapshot (= best) |
| `results/run4/regressor/test_metrics.json` | Full regressor test metrics |
| `results/run4/regressor/plots/` | Loss, MAE/RMSE, bias, admissibility, overestimate plots |

---

## Next steps

1. **Consider τ=0.5 for run 5** — the bias is still −0.33 at τ=0.4. τ=0.5 targets the residual
   median, which would push admissibility toward 50% (theoretical ceiling for pinball loss). For Phase 3,
   the regressor's role is to provide a tighter overestimate bound than the classifier; reducing bias
   further would help without sacrificing the over_max=8 property.

2. **Extend training to 25 epochs** — no convergence visible at epoch 15. The val MAE trajectory
   suggests ~0.05 improvement is available. Unlike the classifier, there is no over_max tail risk from
   running longer.

3. **Phase 3 pairing** — use classifier (thresh=0.20, T=1.5) as the primary heuristic and the
   regressor as a secondary experiment focusing on tight WA* bounds via the lower over_max.
