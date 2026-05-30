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
| 1     | In progress    | Skeleton complete; additive PDB builder is the next step |
| 2     | Not started    | Awaiting Phase 1 dataset                                |
| 3     | Not started    | Awaiting Phase 2 trained model                          |
