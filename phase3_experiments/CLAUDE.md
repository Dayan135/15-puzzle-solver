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

## Architecture — uniform benchmarking

One generic engine + pluggable heuristic adapters. Every run, regardless of
algorithm or heuristic, produces the same `SearchResult` metric set measured in
one place (`search/algorithms.py`): nodes expanded/generated, heuristic
batches/states/time, wall time, termination reason.

```
phase3_experiments/
├── search/
│   ├── puzzle.py        ← vectorized uint64 state ops (phase1 encoding, exact)
│   ├── heuristics.py    ← Heuristic adapters: zero | md | pdb | classifier |
│   │                      regressor | classifier_v2 | regressor_v2 (registry)
│   └── algorithms.py    ← batched best-first engine; WA*/GBFS via
│                          f = g_weight·g + h_weight·h; SearchResult
├── benchmark/
│   ├── runner.py        ← grid driver → CSV (flushed per row, resumable)
│   └── gen_instances.py ← instance CSV generator (uniform / scramble)
├── run_experiments.py   ← YAML-config CLI entry point
├── configs/             ← experiment grids (smoke_local.yaml = template)
├── testset/             ← instance CSVs (id, state_hex, optimal_cost)
├── results/             ← benchmark CSVs
└── tests/               ← 30 pytest cases (engine optimality, PDB rank vs
                           phase1 reference, NN adapter sanity)
```

### Heuristic interface

`h(states: uint64[N]) -> float32[N]` plus `name`/`params()` metadata. New
heuristics register in `HEURISTIC_REGISTRY` and become available to YAML
configs as `{type: <name>, ...kwargs}` with zero engine changes.

### Engine invariants (hard-won, do not regress)

- **h(GOAL) = 0 is pinned by the engine**, never queried from the model. The
  run-2 classifier predicts h(GOAL)=58 (GOAL is the only cost-0 state and fell
  outside its training split); without the pin every search runs to its limit.
- **Goal accepted only when it pops first from OPEN.** A goal popped mid-batch
  is pushed back; otherwise large batches return suboptimal solutions at w=1.
- **batch_size=1 for node-count experiments.** batch>1 amortizes NN forward
  passes but inflates nodes_expanded by up to batch_size× for well-guided
  heuristics (a near-perfect h still expands a full batch per depth level).
  Wall time on MPS/GPU is nearly identical either way; batch_size is recorded
  in every CSV row.

## Running

```bash
cd phase3_experiments        # paths in configs are relative to here

# instances (no ground-truth costs yet — phase1 IDA* oracle pending)
python -m benchmark.gen_instances --n 20 --mode scramble --moves 30 --seed 0 \
    --out testset/smoke.csv

# grid run; reruns skip already-completed rows unless --no-resume
python run_experiments.py --config configs/smoke_local.yaml

# heuristic-quality / search-effort summary table
# (real_avg, h_avg, diff_std, diff_max, exp_avg per heuristic x algorithm)
python -m benchmark.summarize results/smoke.csv [--csv results/summary.csv]

pytest tests/ -q
```

Local dev env: `/opt/anaconda3/envs/chess/bin/python` (torch + MPS).
Cluster: conda env `search` (same as phase2 jobs).

## Output

CSV per row: instance, algorithm+params, heuristic+params, h_start (heuristic
estimate at the start state), solved, termination, solution_cost,
suboptimality, nodes_expanded, nodes_generated, h_batches, h_states, h_time_s,
wall_time_s, batch_size.

`benchmark/summarize.py` aggregates per (heuristic, algorithm): real cost avg,
h(start) avg, std/max of h(start) − real, over_rate (admissibility violations),
found cost avg, suboptimality avg, nodes expanded/generated avg, wall time avg,
h_frac (share of time in the heuristic). True cost = optimal_cost column when
present, else best solution in the file (exact iff an admissible w=1 config ran).

## Status

| Component | Status |
|-----------|--------|
| State core (`puzzle.py`) | ✅ Complete — matches phase1 slide/MD/solvability bit-for-bit (tested vs scalar ports) |
| Heuristic adapters | ✅ Run-2 classifier/regressor + md/zero validated; PDB adapter validated against the real 7-8 tables (copied to `phase1_data_generation/data/pdbs/` locally on 2026-06-11, gitignored; also on cluster); v2 adapters ready, awaiting run-3 checkpoints |
| Search engine (WA*/GBFS) | ✅ Complete — optimality verified vs BFS oracle; batch-invariance regression tests |
| Benchmark runner | ✅ Complete — resumable CSV grid; smoke grid (20×30-move, 12 configs) runs in ~30 s locally |
| Test set w/ optimal costs | ⬜ Next: 1000 uniform states + phase1 IDA*+PDB oracle (needs node counter added to `ida_star.h`) |
| IDA* baseline numbers | ⬜ Blocked on the same node counter |
| Cluster grid + plots | ⬜ After test set; suboptimality-vs-expansions figure |

### Smoke-test signal (20 scrambles × 30 moves, run-2 checkpoints, batch=1)

| heuristic | algo | avg expansions | avg cost (opt=27.2) |
|---|---|---|---|
| pdb (7-8) | A*(w=1) | 103 | 27.2 |
| pdb (7-8) | GBFS | 93 | 45.2 |
| classifier t=0.5 | A*(w=1) | 50 | 27.4 |
| classifier t=0.5 | GBFS | 29 | 28.1 |
| regressor | GBFS | 112 | 45.7 |

Already on shallow states the learned heuristic beats the PDB on expansions
(h_avg 27.0 vs 24.4 against true 27.2); the real experiment is deep uniform
states where the admissible/learned gap widens. PDB rows double as the exact
ground-truth anchor (admissible + w=1).
