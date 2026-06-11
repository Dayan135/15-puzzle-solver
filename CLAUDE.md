# 15-Puzzle Solver — Master's Thesis Project

## Research Goal

Investigate the trade-off between the **accuracy of a learned, non-admissible heuristic** (deep
neural network) and the **computational effort** required to find a satisficing solution using
Weighted A* or GBFS. Domain: 15-Puzzle (4×4 sliding tile puzzle).

The 15-Puzzle was chosen over the Rubik's Cube because IDA* + **additive** pattern databases
solves random states optimally in milliseconds, making it feasible to generate millions of
ground-truth `[state, cost]` pairs within a week on a single CPU machine.

## Three-Phase Pipeline

```
Phase 1                  Phase 2                   Phase 3
─────────────────        ────────────────────────  ─────────────────────────
C++ Data Generator  ───► PyTorch Model Training ──► Search Experiments
                                                    (Weighted A*, GBFS)
  [uint64_t state,         Learns h_θ(s) ≈ cost*    Measures: nodes expanded
   uint8_t cost]           (non-admissible ok)       vs. solution quality
  pairs on disk
```

## Directory Layout

```
15-puzzle-solver/
├── CLAUDE.md                    ← this file
├── phase1_data_generation/      ← C++ IDA* pipeline
│   ├── CLAUDE.md
│   ├── CMakeLists.txt
│   ├── include/
│   ├── src/
│   └── tests/
├── phase2_model_training/       ← PyTorch supervised learning (future)
│   └── CLAUDE.md
└── phase3_experiments/          ← Search evaluation harness (future)
    └── CLAUDE.md
```

## Shared Data Contract

Binary dataset files written by Phase 1, read by Phase 2 & 3.

**File format** (written by Phase 1):
```
Header  (16 bytes): magic=0x50313544 ("P15D")  version=1  n_records (uint64)
Record   (9 bytes): state (uint64 LE, nibble-packed board) + cost (uint8)
Files split at 1 GB → dataset_000.bin, dataset_001.bin, ...
```

**State encoding** — one `uint64_t`, 16 nibbles:
```
nibble p (bits 4p..4p+3) = tile at cell p   (tile 0 = blank)
Goal = 0xFEDCBA9876543210   (tile p at cell p; blank at cell 0)
```

**Reading in Python** (Phase 2 / Phase 3):
```python
import struct

def read_dataset(path):
    with open(path, 'rb') as f:
        magic, version = struct.unpack('<II', f.read(8))
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            state, cost = struct.unpack('<QB', f.read(9))
            yield state, cost
```

## Hardware

- CPU: Intel Core i7-11700F (8 cores / 16 threads)
- OS: Windows 11, MinGW-w64 / GCC toolchain
- Dev: macOS (Apple Silicon), clang++

## Status

| Phase | Status         | Notes                                                   |
|-------|----------------|---------------------------------------------------------|
| 1     | ✅ Complete     | 100M `[state, cost]` pairs generated on the cluster (job 17947513, 2026-05-31). Additive 7-8 PDBs + IDA* + stratified buckets. Output: `data/full/dataset_000.bin` (900 MB) on the cluster. |
| 2     | In progress    | Run 2 complete. Classifier: MAE=0.996, admissibility=76.7% (job 18018359, 14 ep). Regressor: MAE=1.287, admissibility=82.4% (job 18018360, 10 ep). Threshold sweep done (job 18037374): best ≥99% admissibility is thresh=0.10 temp=2.0 (MAE=2.301, over_max=23). Run 3 implementation complete (272-dim input + residual target + chunked init); awaiting cluster training. |
| 3     | In progress    | Benchmark infrastructure complete (2026-06-11): generic batched best-first engine (WA*/GBFS), heuristic adapter registry, resumable CSV grid runner; validated end-to-end with run-2 checkpoints. Next: test set with phase1 IDA* oracle costs. Run-2 checkpoints live in `phase2_model_training/checkpoints/` (local + cluster, gitignored). |

### Phase 1 dataset (cluster)

- **Location**: `phase1_data_generation/data/full/dataset_000.bin` on `slurm.bgu.ac.il` (gitignored; not in repo).
- **Size**: 100,000,000 records, 900,000,016 bytes (header verified `n_records=100M`).
- **PDBs**: `data/pdbs/{pdb_a.bin (55 MB), pdb_b.bin (495 MB)}` — built once (job 17947419), reused.
- **Distribution**: stratified across 22 scramble-length buckets (~4.5M each), mean cost spanning 1 → 52.55.
- **Duplicate handling for Phase 2**: 100M records are only ~42% distinct — shallow buckets repeat a few near-goal states millions of times. Two mechanisms work together:
  - `train_val_test_split` partitions on *unique states* (not records): val/test receive exactly one record per unique state, so no training state leaks into evaluation.
  - `CostBalancedSampler` draws each cost value with equal probability during training, making duplicate shallow records statistically weightless. Train retains all duplicate records (they help the sampler cover low-cost values).
