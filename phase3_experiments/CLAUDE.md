# Phase 3 — Experiments

## Goal

Empirically measure the **trade-off between solution quality and search effort** when replacing the admissible PDB heuristic with the learned non-admissible heuristic `h_θ(s)` from Phase 2.

Two search algorithms under study:
- **Weighted A\*** (WA*) with weight `w ≥ 1`: finds solutions of cost ≤ `w × optimal`
- **Greedy Best-First Search** (GBFS): ignores path cost entirely, purely heuristic-guided

## Research Questions

1. How does solution suboptimality scale with `w` in WA*?
2. How many nodes does each algorithm expand compared to optimal IDA*?
3. At what `w` does WA* with `h_θ` match IDA* with PDBs in speed while staying near-optimal?
4. Does GBFS with `h_θ` find solutions faster than IDA* even if they're not optimal?

## Experimental Design

### Independent variables
- Search algorithm: IDA* (baseline), WA* (w = 1.0, 1.1, 1.5, 2.0, 3.0, ∞), GBFS
- Heuristic: PDB (admissible), `h_θ` (learned, non-admissible)

### Dependent variables
- Nodes expanded per problem instance
- Solution cost (number of moves)
- Wall-clock solve time
- Suboptimality ratio: `found_cost / optimal_cost`

### Test set
- 1000 uniformly random states (separate from the training set)
- Ground-truth optimal costs available from Phase 1's IDA* solver

## Implementation Plan (TBD)

The search harness will likely be Python (calling the Phase 2 model for `h_θ`) with
the state transition logic either:
- Pure Python (simple, slower)
- C extension wrapping Phase 1's `slide` / neighbor tables (fast, reuses existing code)

The Phase 1 C++ `IDAStar<SumHeuristic>` solver serves as the **baseline** and ground-truth oracle.

## Output

- CSV tables of `[algorithm, w, state_id, nodes_expanded, solution_cost, time_ms]`
- Plots: suboptimality vs. nodes expanded (the core thesis figure)

## Status

Not started — awaiting Phase 2 trained model.
