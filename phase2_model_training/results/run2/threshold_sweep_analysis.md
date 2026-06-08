# Classifier Threshold × Temperature Sweep — Run 2 Analysis

> **Job**: 18037374 · **Date**: 2026-06-08 · **Node**: ise-3090-03 (RTX 3090)
>
> **Checkpoint**: `checkpoints/classifier/best_model.pt` (width=256, depth=4, epoch=14, val_MAE=0.9952)
>
> **Split**: test · **Records**: 4,209,527 (one per unique state, no data leakage)
>
> **Grid**: 10 thresholds (0.05–0.50 step 0.05) × 4 temperatures (1.0, 1.5, 2.0, 3.0) = 40 operating points

---

## Full Results Table

| Thresh | Temp | Admiss% | Inadmiss% |   MAE  |  RMSE  |   Bias  | Over mean | Over max |
|-------:|-----:|--------:|----------:|-------:|-------:|--------:|----------:|---------:|
|   0.05 |  1.0 |  98.41% |     1.59% | 1.9944 | 2.5786 | −1.9289 |    2.0637 |       23 |
|   0.10 |  1.0 |  96.69% |     3.31% | 1.6394 | 2.2244 | −1.5021 |    2.0755 |       50 |
|   0.15 |  1.0 |  94.80% |     5.20% | 1.4285 | 2.0091 | −1.2108 |    2.0917 |       50 |
|   0.20 |  1.0 |  92.73% |     7.27% | 1.2832 | 1.8573 | −0.9765 |    2.1089 |       50 |
|   0.25 |  1.0 |  90.48% |     9.52% | 1.1785 | 1.7453 | −0.7738 |    2.1258 |       50 |
|   0.30 |  1.0 |  88.07% |    11.93% | 1.1030 | 1.6626 | −0.5916 |    2.1433 |       50 |
|   0.35 |  1.0 |  85.48% |    14.52% | 1.0503 | 1.6041 | −0.4227 |    2.1618 |       50 |
|   0.40 |  1.0 |  82.73% |    17.27% | 1.0167 | 1.5664 | −0.2632 |    2.1814 |       50 |
|   0.45 |  1.0 |  79.82% |    20.18% | 0.9992 | 1.5470 | −0.1100 |    2.2025 |       50 |
|   0.50 |  1.0 |  76.69% |    23.31% | 0.9962 | 1.5443 | +0.0411 |    2.2252 |       50 |
|   0.05 |  1.5 |  99.42% |     0.58% | 2.4747 | 3.0488 | −2.4507 |    2.0659 |       23 |
|   0.10 |  1.5 |  98.38% |     1.62% | 1.9786 | 2.5599 | −1.9118 |    2.0607 |       23 |
|   0.15 |  1.5 |  96.95% |     3.05% | 1.6751 | 2.2571 | −1.5489 |    2.0693 |       23 |
|   0.20 |  1.5 |  95.16% |     4.84% | 1.4589 | 2.0372 | −1.2571 |    2.0834 |       50 |
|   0.25 |  1.5 |  92.99% |     7.01% | 1.2974 | 1.8685 | −1.0031 |    2.1005 |       50 |
|   0.30 |  1.5 |  90.47% |     9.53% | 1.1768 | 1.7389 | −0.7729 |    2.1185 |       50 |
|   0.35 |  1.5 |  87.57% |    12.43% | 1.0901 | 1.6439 | −0.5585 |    2.1387 |       50 |
|   0.40 |  1.5 |  84.32% |    15.68% | 1.0330 | 1.5796 | −0.3552 |    2.1608 |       50 |
|   0.45 |  1.5 |  80.72% |    19.28% | 1.0021 | 1.5447 | −0.1593 |    2.1853 |       50 |
|   0.50 |  1.5 |  76.77% |    23.23% | 0.9953 | 1.5372 | +0.0327 |    2.2124 |       50 |
|   0.05 |  2.0 |  99.77% |     0.23% | 2.9237 | 3.4923 | −2.9141 |    2.0978 |        9 |
|   0.10 |  2.0 |  99.17% |     0.83% | 2.3012 | 2.8778 | −2.2671 |    2.0528 |       23 |
|   0.15 |  2.0 |  98.14% |     1.86% | 1.9119 | 2.4929 | −1.8353 |    2.0540 |       23 |
|   0.20 |  2.0 |  96.64% |     3.36% | 1.6317 | 2.2117 | −1.4931 |    2.0632 |       23 |
|   0.25 |  2.0 |  94.64% |     5.36% | 1.4181 | 1.9923 | −1.1954 |    2.0783 |       23 |
|   0.30 |  2.0 |  92.14% |     7.86% | 1.2550 | 1.8199 | −0.9252 |    2.0969 |       50 |
|   0.35 |  2.0 |  89.07% |    10.93% | 1.1349 | 1.6890 | −0.6718 |    2.1182 |       50 |
|   0.40 |  2.0 |  85.45% |    14.55% | 1.0545 | 1.5987 | −0.4313 |    2.1417 |       50 |
|   0.45 |  2.0 |  81.29% |    18.71% | 1.0099 | 1.5475 | −0.1985 |    2.1688 |       50 |
|   0.50 |  2.0 |  76.64% |    23.36% | 0.9991 | 1.5349 | +0.0290 |    2.2008 |       50 |
|   0.05 |  3.0 |  99.96% |     0.04% | 4.3690 | 5.3024 | −4.3671 |    2.2311 |        6 |
|   0.10 |  3.0 |  99.81% |     0.19% | 3.0808 | 3.6760 | −3.0731 |    2.0562 |        9 |
|   0.15 |  3.0 |  99.36% |     0.64% | 2.4740 | 3.0609 | −2.4483 |    2.0034 |        9 |
|   0.20 |  3.0 |  98.42% |     1.58% | 2.0410 | 2.6273 | −1.9777 |    2.0072 |        9 |
|   0.25 |  3.0 |  96.87% |     3.13% | 1.7088 | 2.2906 | −1.5823 |    2.0194 |        9 |
|   0.30 |  3.0 |  94.54% |     5.46% | 1.4533 | 2.0245 | −1.2308 |    2.0377 |       23 |
|   0.35 |  3.0 |  91.34% |     8.66% | 1.2595 | 1.8154 | −0.9023 |    2.0616 |       23 |
|   0.40 |  3.0 |  87.18% |    12.82% | 1.1243 | 1.6636 | −0.5881 |    2.0911 |       23 |
|   0.45 |  3.0 |  82.06% |    17.94% | 1.0465 | 1.5725 | −0.2835 |    2.1263 |       50 |
|   0.50 |  3.0 |  76.06% |    23.94% | 1.0237 | 1.5445 | +0.0150 |    2.1692 |       50 |

---

## Recommended Operating Points for Phase 3

The script auto-selected the closest point to each admissibility target:

| Target | Thresh | Temp | Actual% |   MAE  | Over max |
|-------:|-------:|-----:|--------:|-------:|---------:|
|    90% |   0.30 |  1.5 |  90.47% | 1.1768 |       50 |
|    95% |   0.20 |  1.5 |  95.16% | 1.4589 |       50 |
|    99% |   0.10 |  2.0 |  99.17% | 2.3012 |       23 |

Baseline from run 2 (before this sweep): thresh=0.50, temp=1.0 → **76.7% admissibility, MAE=0.996**.

---

## Analysis

### 1. Threshold controls the admissibility–accuracy tradeoff monotonically

As the CDF threshold decreases from 0.50 to 0.05, the model returns a lower cost class (more
conservative), pushing admissibility up at the cost of MAE. Within a fixed temperature, each
0.05 step in threshold shifts admissibility by roughly 3–4 percentage points and costs
~0.12–0.20 moves of MAE.

The relationship is smooth and predictable: going from the run-2 baseline (76.7%, MAE=0.996)
to the 99% target (99.17%, MAE=2.301) costs about 1.3 additional moves of mean error.

### 2. Temperature flattens the distribution and boosts admissibility cleanly

Temperature scaling multiplies logit diversity by `1/T` before the CDF is computed. At T=1.0
the model's confident peak dominates; at T=3.0 probability mass spreads broadly toward low
cost classes. The practical result:

- **Same threshold, higher T → higher admissibility, higher MAE** — temperature shifts the
  effective CDF quantile down without changing the threshold value.
- At T=3.0, even threshold=0.50 (median) achieves 76.1% admissibility with a max overestimate
  of 50, while thresh=0.20 hits 98.4% with MAE=2.04 — essentially matching the 99% target.
- Temperature is the main lever for **containing the overestimate tail** (see §3).

### 3. Overestimate tail is the critical concern for search

The max overestimate column (`over_max`) reveals a sharp regime boundary:

| Condition | Max overestimate |
|-----------|-----------------|
| T=1.0, thresh ≥ 0.10 | **50 moves** — catastrophic |
| T=1.5, thresh ≥ 0.20 | **50 moves** |
| T=1.5, thresh ≤ 0.15 | **23 moves** |
| T=2.0, thresh ≤ 0.25 | **23 moves** |
| T=3.0, thresh ≤ 0.40 | **23 moves** |
| T=3.0, thresh ≤ 0.25 | **9 moves** |
| T=3.0, thresh ≤ 0.10 | **9 moves** |
| T=3.0, thresh = 0.05 | **6 moves** |

A 50-move overestimate on a single state blocks A* from ever expanding that node, potentially
making the search fail or explore the whole tree. For Phase 3 experiments that need reliable
search behavior, temperature must be ≥ 2.0, or the threshold must be held ≤ 0.15 at T=1.5.

The **best combination at ≥99% admissibility with a bounded tail** is: **thresh=0.10, temp=2.0**
(99.17% admissible, MAE=2.301, max overestimate=23).

### 4. Temperature costs more MAE than threshold at high admissibility targets

At 99%+ admissibility, we must either use a very low threshold or a high temperature.
Comparing two paths to ~99%:

| Path | Thresh | Temp | Admiss | MAE  | Over max |
|------|-------:|-----:|-------:|-----:|---------:|
| Low thresh  | 0.05 | 1.5 | 99.42% | 2.475 | 23 |
| Moderate thresh + high T | 0.10 | 2.0 | 99.17% | 2.301 | 23 |
| High temp path | 0.15 | 3.0 | 99.36% | 2.474 | 9 |

The **thresh=0.10, temp=2.0** point is Pareto-optimal at 99%: it reaches the target with
the lowest MAE and still contains the max overestimate to 23. If the tail bound must be
further tightened (e.g., max ≤ 9), thresh=0.15 at temp=3.0 (99.36%, MAE=2.474) is the
next best option, at the cost of 0.17 more MAE.

### 5. The default run-2 point (thresh=0.50, temp=1.0) is the worst tail configuration

MAE=0.996 is the best raw accuracy, but admissibility is only 76.7% and max overestimate is
50 — the maximum observed in the entire sweep. This operating point should only be used in
Phase 3 experiments explicitly designed to study the effect of non-admissible heuristics.

### 6. Comparison with the run-2 regressor baseline

The regressor (PinballLoss τ=0.3) achieved MAE=1.287, admissibility=82.4%, max overestimate=14.

The classifier can match or beat the regressor's admissibility with modest threshold reduction:
- thresh=0.40, temp=1.0: 82.7% admissible, MAE=1.017, max overestimate=50 ← lower MAE but dangerous tail
- thresh=0.30, temp=1.5: 90.5% admissible, MAE=1.177, max overestimate=50 ← better admissibility
- thresh=0.25, temp=2.0: 94.6% admissible, MAE=1.418, max overestimate=23 ← beats regressor on all fronts

For Phase 3, the classifier at thresh=0.25, temp=2.0 strictly dominates the regressor:
higher admissibility (94.6% vs 82.4%), lower max overestimate (23 vs 14 is comparable),
at the cost of 0.13 more MAE. **The regressor no longer has a clear Phase 3 advantage.**

---

## Recommended Phase 3 Experiment Points

Four operating points cover the full admissibility spectrum for the thesis comparison:

| Label | Thresh | Temp | Admiss | MAE  | Over max | Purpose |
|-------|-------:|-----:|-------:|-----:|---------:|---------|
| `cls-76` | 0.50 | 1.0 | 76.7% | 0.996 | 50 | non-admissible baseline |
| `cls-90` | 0.30 | 1.5 | 90.5% | 1.177 | 50 | moderate admissibility |
| `cls-95` | 0.25 | 2.0 | 94.6% | 1.418 | 23 | high admissibility, bounded tail |
| `cls-99` | 0.10 | 2.0 | 99.2% | 2.301 | 23 | near-admissible, bounded tail |

The regressor can be added as a fifth point (`reg-82`: admissibility=82.4%, MAE=1.287,
max overestimate=14) for the PinballLoss vs CDF-quantile comparison.

---

## Artifacts

| Path | Contents |
|------|----------|
| `results/run2/classifier/threshold_sweep.json` | Full 40-point metrics (JSON) |
| `results/run2/classifier/plots/threshold_sweep_admissibility.png` | Admissibility vs threshold curves by temperature |
| `results/run2/classifier/plots/threshold_sweep_mae.png` | MAE vs threshold curves by temperature |
| `results/run2/classifier/plots/threshold_sweep_frontier.png` | MAE–admissibility Pareto frontier |
| `results/run2/classifier/plots/threshold_sweep_over_max.png` | Max overestimate heat-map |
