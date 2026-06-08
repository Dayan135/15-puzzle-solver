# Phase 2 Training — Run 1 Analysis

> **⚠️ These results are invalid for final evaluation.**
> The train/val/test split had a data pipeline bug: duplicate records leaked into the val and test sets,
> so reported metrics are optimistically biased. A corrected run is required before drawing conclusions
> about model quality. The observations below are directional only.

---

## Run info

Jobs 17975849 (classifier) and 17975850 (regressor), 2026-06-03, 10 epochs each, CPU-only nodes (~9 h/job).
Dataset: 100M records, 80/10/10 split (contaminated).

---

## Results snapshot

| Metric | Classifier | Regressor |
|---|---|---|
| MAE | **0.431** | 0.586 |
| RMSE | **1.022** | 1.169 |
| Mean signed error | +0.012 (neutral) | −0.337 (underestimates) |
| Admissibility rate | 90.1% | **91.5%** |
| Overestimate mean | 2.24 moves | **1.45 moves** |
| Overestimate max | **42 moves** ⚠️ | **8 moves** |
| Params | 614,993 | 594,433 |

---

## What we can learn

### 1. Classifier inference: use CDF quantile, not argmax

The argmax strategy produced a **42-move max overestimate**. A single prediction that far above true cost
causes WA* to deeply mis-rank a node and can force thousands of extra expansions to recover.

**Proposed fix:** at inference, accumulate softmax probabilities from cost 0 upward and return the smallest
`y` such that `P(cost ≤ y) ≥ threshold`. This returns a quantile of the predicted distribution instead
of its mode.

Why this is better:
- Gives a **tunable admissibility dial** without retraining — sweep threshold on val set, pick the floor you want.
- Eliminates catastrophic tail overestimates: even if the model puts the mode at the wrong class, the CDF
  reaches the threshold well before the outlier region.
- Strictly more flexible than the regressor's PinballLoss approach, which commits to a fixed quantile at
  training time. One trained classifier covers all quantiles.

**Implementation note:** the threshold does not equal admissibility rate without calibration — CrossEntropyLoss
does not produce calibrated probabilities. Measure actual admissibility at each threshold on the val set;
don't assume them equal.

---

### 2. Prefer full admissibility over low MAE

The regressor (91.5% admissible, MAE 0.586) is more useful for WA* than the classifier (90.1% admissible,
MAE 0.431) despite worse average accuracy. The reason is structural:

- WA*(w) with an admissible heuristic guarantees `found_cost ≤ w × optimal`. This bound is what makes the
  Phase 3 suboptimality curves interpretable and meaningful.
- With an inadmissible heuristic, that bound is gone. Measured suboptimality ratios become a function of
  which states happened to be in the test set, not a clean function of w.
- A 42-move overestimate is not a statistical nuisance — it breaks the search's ability to reason about
  solution quality for that node.

**Target for future runs:** admissibility rate ≥ 99%, with MAE as a secondary objective.
The CDF inference approach on the classifier is the most direct path to achieving both.

For GBFS experiments, admissibility is irrelevant (GBFS ignores path cost), so the classifier's
raw accuracy advantage matters more there.

---

### 3. Classifier is still learning at epoch 10

MAE trajectory: 0.519 → 0.488 → 0.473 → 0.465 → 0.459 → 0.452 → 0.445 → 0.440 → 0.434 → 0.431 — no plateau.
The corrected run should target ≥ 20 epochs for the classifier.

The regressor plateaued around epoch 5–6 and showed noisy, flat improvement after that. 10 epochs is likely enough.

---

### 4. Both nodes ran without GPU

Both second-run jobs landed on CPU-only nodes (~9 h each). Once the data bug is fixed and the corrected run
is submitted, prioritize GPU allocation — the first regressor run (job 17966278, dt-3090-01, RTX 3090)
ran significantly faster.
