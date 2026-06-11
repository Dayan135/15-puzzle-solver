# Phase 2 — Run 3 Improvement Suggestions

> Based on run 2 results (jobs 18018359/18018360, 2026-06-08).
> These suggestions are ordered by expected impact and implementation cost.

---

## Summary of issues in run 2

| Issue | Classifier | Regressor |
|---|---|---|
| Admissibility far from ≥99% target | 76.7% at threshold=0.5 | 82.4% |
| Overestimate max too large for safe search | 50 moves | 14 moves |
| Both models still learning at final epoch | ✗ plateau at ep 14 | ✗ plateau at ep 10 |
| Input carries no spatial geometry | one-hot only | one-hot only |
| Target variance is high | predicting raw h*(s) | predicting raw h*(s) |

---

## Suggestion 1 — Threshold sweep (no retraining, do now)

**What:** Evaluate the saved run 2 classifier checkpoint on the val/test split across a grid of
`cdf_threshold ∈ [0.05, 0.10, ..., 0.50]` and `temperature ∈ [1.0, 1.5, 2.0, 3.0]`.

**Why it matters:** Threshold and temperature are pure inference-time parameters. No GPU job needed.
This maps the full admissibility/MAE tradeoff curve and answers: "at what threshold does the classifier
reach 95% or 99% admissibility, and what is the MAE cost?"

**How:** Run `evaluate_thresholds.py` on the existing checkpoint:
```bash
cd phase2_model_training
python evaluate_thresholds.py \
    --checkpoint checkpoints/classifier/best_model.pt \
    --data /path/to/dataset_000.bin
```

**Expected output:** A table like:

```
threshold  temp   admiss%   MAE    over_mean  over_max
  0.05     1.0    99.X%    1.XX   X.XX       X
  0.10     1.0    98.X%    1.XX   ...
  ...
  0.50     1.0    76.7%    0.996  2.23       50
  0.05     1.5    ...
```

**Action after sweep:** Use the resulting curve to pick operating points for Phase 3 experiments.
Document the threshold used in Phase 3 configs.

---

## Suggestion 2 — Add per-tile Manhattan distances to input (requires retraining)

**What:** Extend the 256-dim one-hot input to 272 dims by appending 16 Manhattan distance features —
one per cell, equal to the Manhattan distance of the tile currently at that cell from its goal cell.

**Why it matters:** The one-hot encoding tells the model *which tile is where* but gives no geometric
signal about *how far from goal*. The sum of Manhattan distances is the strongest admissible lower
bound on h*(s), and the per-tile breakdown tells the model exactly which tiles are contributing most.
The network currently has to discover spatial structure from position indices alone; providing distances
directly eliminates this bottleneck.

**Implementation — `data/puzzle_dataset.py` → `PuzzleDataset.__getitem__`:**

```python
def __getitem__(self, idx: int):
    tiles = self.nibbles[idx]              # uint8[16]: tile at each cell
    x = np.zeros(272, dtype=np.float32)
    x[_CELL_OFFSETS + tiles] = 1.0        # one-hot: cells 0–255

    # Per-tile Manhattan distance from current cell to goal cell
    cur_row = _CELL_OFFSETS // 16 // 4    # precompute as module constant
    cur_col = _CELL_OFFSETS // 16 % 4
    goal_row = tiles.astype(np.int32) // 4
    goal_col = tiles.astype(np.int32) % 4
    x[256:272] = np.abs(cur_row - goal_row) + np.abs(cur_col - goal_col)
    # tile 0 (blank) has goal at cell 0 → distance = row + col of blank cell (correct)

    cost = float(self.costs[idx])
    y    = cost / MAX_COST if self.normalize_cost else cost
    return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
```

Cleaner version using pre-declared module-level constants (put at module top):
```python
_CELL_ROW = np.arange(16, dtype=np.int32) // 4   # row of each cell
_CELL_COL = np.arange(16, dtype=np.int32) % 4    # col of each cell
```

Then in `__getitem__`:
```python
goal_row = tiles.astype(np.int32) // 4
goal_col = tiles.astype(np.int32) % 4
dists    = np.abs(_CELL_ROW - goal_row) + np.abs(_CELL_COL - goal_col)  # [16]
x[256:272] = dists.astype(np.float32)
```

**Model change — `model/classifier.py` and `model/regressor.py`:**

Change the projection layer input from 256 to 272:
```python
self.proj = nn.Sequential(nn.Linear(272, width), nn.ReLU())
```

**Config change:** Add `input_dim: 272` to `configs/classifier.yaml` and `configs/regressor.yaml`,
pass it through `_build_model()` in `train.py`. Or simply hardcode 272 if only one input mode is used.

**Note:** The Manhattan distance sum is already ≤ h*(s) (admissible lower bound). Providing it as
input does not give the model the answer — it tells the model the floor so it can predict the gap.

---

## Suggestion 3 — Predict residual h*(s) − MD(s) as target (requires retraining)

**What:** Instead of predicting h*(s) directly, train both models to predict the *residual*:
`r(s) = h*(s) − MD_sum(s)`, where `MD_sum(s) = Σ manhattan_distance(tile_at_cell_p, goal_of_tile_at_cell_p)`.

**Why it matters:**
- MD_sum(s) ≤ h*(s) always, so r(s) ≥ 0 always. The model output space becomes [0, ~35]
  instead of [0, 80], dramatically reducing variance.
- The model only needs to learn the correction above the Manhattan lower bound — a much
  simpler function than h*(s) itself.
- For the classifier, `NUM_COSTS` can be reduced from 81 to ~40, making the classification
  task easier and the CDF inference more precise (fewer bins to misplace mass across).
- Overestimate max will shrink: the residual rarely exceeds 30 moves, so tail errors are bounded.

**Implementation — `data/puzzle_dataset.py`:**

Add `residual_target: bool = False` parameter to `PuzzleDataset.__init__`. When True:
```python
# In __getitem__:
md_sum = int((np.abs(_CELL_ROW - goal_row) + np.abs(_CELL_COL - goal_col)).sum())
# exclude blank tile (tile 0) from MD sum — standard Manhattan heuristic
blank_pos = np.where(tiles == 0)[0][0]
md_sum -= np.abs(_CELL_ROW[blank_pos] - 0) + np.abs(_CELL_COL[blank_pos] - 0)
residual = float(self.costs[idx]) - md_sum
y = residual  # or residual / MAX_RESIDUAL if normalizing
```

**Phase 3 reconstruction:**  
When computing heuristic values in the search loop, restore h*(s):
```python
h_theta = model.predict(x) + manhattan_sum(state)
```
This must be done consistently everywhere `predict()` is called in Phase 3.

**Recommended combination:** Suggestion 2 + Suggestion 3 together — input includes MD features,
target is the residual above MD. The model receives the lower bound as input and predicts the gap.

---

## Suggestion 4 — Temperature scaling at inference (no retraining)

**What:** Divide logits by temperature T > 1 before the softmax in `predict_quantile`:
```python
cdf = torch.softmax(self.forward(x) / temperature, dim=-1).cumsum(dim=-1)
```

**Why it matters:** CrossEntropyLoss produces overconfident probability distributions —
the model often puts most mass on a single cost class. When that class is wrong, the CDF
crosses the threshold at that wrong class, giving a large overestimate. Temperature T > 1
spreads the distribution, so the CDF is smoother and the quantile inference is more robust
to single-bin overconfidence.

This is the cause of `overestimate_max=50` at threshold=0.5 — the model is extremely confident
about one high-cost bin for some states. Temperature scaling would reduce this.

**No retraining required.** Sweep temperature in `evaluate_thresholds.py` to find the best value.
Good starting range: T ∈ [1.0, 1.5, 2.0, 3.0].

**Add to `classifier.py`:**
```python
@torch.no_grad()
def predict_quantile(self, x, threshold=0.5, temperature=1.0):
    cdf = torch.softmax(self.forward(x) / temperature, dim=-1).cumsum(dim=-1)
    return (cdf < threshold).sum(dim=-1).clamp(max=80).long()
```

---

## Suggestion 5 — More epochs

**Classifier:** Still declining at epoch 14 (loss 0.7361 → 0.5409, MAE 1.174 → 0.995).
Target 20–25 epochs in run 3 if feature engineering changes are not applied, or 14+ with them.

**Regressor:** Less clear plateau — drops 1.441 → 1.286 with some oscillation. Try 15 epochs.

---

## Suggested run 3 configuration

### If running a quick benchmark (no feature engineering):
- Classifier only, 25 epochs, `cdf_threshold=None` (log argmax too, for reference)
- Focus is on getting more of the admissibility/MAE curve from the saved checkpoint sweep first

### If running with feature engineering (recommended):
| Change | Classifier | Regressor |
|---|---|---|
| Input dim | 256 → 272 (add MD) | 256 → 272 (add MD) |
| Target | h*(s) − MD_sum(s) | h*(s) − MD_sum(s) |
| NUM_COSTS | 81 → 45 (residual fits 0–44) | n/a |
| Epochs | 20 | 15 |
| cdf_threshold | 0.5 (still, sweep post-hoc) | n/a |
| tau | n/a | 0.3 (unchanged) |

**Expected improvements:**
- Lower MAE from reduced prediction variance
- Lower overestimate_max (residual is bounded)
- Faster convergence (smaller output space)
- More calibrated CDF (fewer bins with noise)

---

## Do NOT change

- `CostBalancedSampler` — still needed to flatten duplicated shallow buckets
- `train_val_test_split` unique-state logic — critical for leak-free evaluation
- AdamW + CosineAnnealingLR — working well, no evidence they are the bottleneck
- Batch size 1024 — GPU utilization is at ~70 k samp/s for classifier; fine
- Architecture depth/width (256, depth=4) — 614 k params is appropriate for the task size
