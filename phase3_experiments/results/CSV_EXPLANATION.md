# Benchmark result files

Two kinds of CSV live here. Per-run files (e.g. `smoke.csv`) are written by
`run_experiments.py` — one row per (instance, algorithm, heuristic) search.
Summary files (e.g. `smoke_summary.csv`) are written by
`python -m benchmark.summarize <per-run.csv> --csv <summary.csv>` — one row
per (heuristic, algorithm) group, aggregated over instances.

All searches in one file ran the identical instance set, so rows are paired:
differences between groups come from the heuristic/algorithm, not the
instance draw.

## Per-run CSV (one row = one search)

| column | meaning |
|---|---|
| `instance_id` | Index into the instance file (`testset/*.csv`). Same id = same start state across all rows. |
| `state_hex` | Start state as 16 hex nibbles (nibble *p* = tile at cell *p*, `0` = blank; goal is `fedcba9876543210`). |
| `optimal_cost` | Ground-truth optimal solution cost from the instance file. Empty if no oracle cost exists yet. |
| `algorithm` | `wastar` (f = g + w·h) or `gbfs` (f = h). |
| `algo_params` | JSON, e.g. `{"w": 2.0}`. Empty `{}` for gbfs. |
| `heuristic` | Adapter name: `pdb`, `md`, `zero`, `classifier`, `regressor`, `classifier_v2`, `regressor_v2`. |
| `heuristic_params` | JSON of the adapter's settings (checkpoint path, threshold, temperature, device, ...). |
| `h_start` | Heuristic estimate at the start state, h(s₀). Compare with `optimal_cost` for heuristic accuracy. |
| `solved` | `1` if a solution was found within the limits, else `0`. |
| `termination` | Why the search stopped: `solved`, `node_limit` (hit max_expansions), `time_limit` (hit max_time_s), `exhausted` (OPEN emptied — cannot happen on solvable inputs). |
| `solution_cost` | Number of moves in the found solution. Empty when unsolved. |
| `suboptimality` | `solution_cost / optimal_cost`. Empty when unsolved or `optimal_cost` is missing. 1.0 = optimal. |
| `nodes_expanded` | Nodes popped from OPEN and expanded (successors generated). The primary search-effort metric. |
| `nodes_generated` | Successor states created (≈ 2–4× expanded; counts duplicates before dedup). |
| `h_batches` | Heuristic invocations (forward passes / vectorized calls). |
| `h_states` | Unique states sent to the heuristic (cache misses; each state is evaluated at most once per run). |
| `h_time_s` | Wall seconds spent inside the heuristic. |
| `wall_time_s` | Total wall seconds for the search. |
| `batch_size` | Engine expansion batch. **1 = faithful node counts.** Larger values amortize NN calls but inflate `nodes_expanded` by up to batch_size× for well-guided heuristics — only compare rows with equal batch_size. |

## Summary CSV (one row = one heuristic × algorithm group)

True cost per instance = `optimal_cost` when present, else the best
`solution_cost` found for that instance anywhere in the per-run file (exact
iff the grid contains an admissible w=1 config, e.g. pdb + wastar(w=1.0);
the tool prints a note when this fallback is used).

| column | meaning |
|---|---|
| `heuristic` | Adapter plus its identity params, e.g. `classifier(temperature=2.0,threshold=0.1)`. Paths/device are omitted. |
| `algorithm` | `wastar(w=...)` or `gbfs`. |
| `n` | Runs in the group (= number of instances). |
| `solved` | How many of the n runs found a solution. All other columns average over the rows indicated below. |
| `real_avg` | Mean true cost over instances. Identical across rows of the same file by construction. |
| `h_avg` | Mean `h_start`. `h_avg − real_avg` is the heuristic's bias at the start states. |
| `diff_std` | Std of `h_start − real` (sample std over instances). |
| `diff_max` | Max of `h_start − real`: the worst overestimation. ≤ 0 means never overestimated on this set. |
| `over_rate` | Fraction of instances with `h_start > real` — measured admissibility-violation rate. |
| `cost_avg` | Mean `solution_cost` over solved runs. |
| `subopt_avg` | Mean of per-instance `solution_cost / real` over solved runs with known true cost. 1.000 = always optimal. |
| `exp_avg` | Mean `nodes_expanded` over all runs in the group (including unsolved — they paid their effort). |
| `gen_avg` | Mean `nodes_generated`, same population as `exp_avg`. |
| `time_avg` | Mean `wall_time_s` per run. Wall time is machine-dependent — prefer `exp_avg` for cross-machine comparisons. |
| `h_frac` | Σ`h_time_s` / Σ`wall_time_s` for the group: share of runtime spent inside the heuristic (≈98% for NN adapters, ≈10–80% for table lookups). |

Note: the five accuracy columns (`real_avg`, `h_avg`, `diff_std`, `diff_max`,
`over_rate`) depend only on the heuristic, so they repeat across the
algorithm rows of the same heuristic. The effort/quality columns are what
vary per algorithm.
