# Rubik's Cube Solver — Master's Thesis Project

## Research Goal

Investigate the trade-off between the **accuracy of a learned, non-admissible heuristic** (deep neural network) and the **computational effort** required to find a satisficing solution using Weighted A* or GBFS. Domain: 3×3×3 Rubik's Cube.

## Three-Phase Pipeline

```
Phase 1                  Phase 2                   Phase 3
─────────────────        ────────────────────────  ─────────────────────────
C++ Data Generator  ───► PyTorch Model Training ──► Search Experiments
                                                    (Weighted A*, GBFS)
  100M [State,Cost]        Learns h(s) ≈ cost*        Measures: nodes expanded
  pairs on disk            (non-admissible)            vs. solution suboptimality
```

## Directory Layout

```
Rubik-s-cube-solver/
├── CLAUDE.md                    ← this file (project-wide guide)
├── phase1_data_generation/      ← C++ IDA* pipeline (complete)
│   ├── CLAUDE.md
│   ├── CMakeLists.txt
│   ├── include/
│   └── src/
├── phase2_model_training/       ← PyTorch supervised learning (future)
│   └── CLAUDE.md
└── phase3_experiments/          ← Search evaluation harness (future)
    └── CLAUDE.md
```

## Shared Data Contract

All phases communicate through binary dataset files at `D:\Search_DB\`.

**File format** (written by Phase 1, read by Phase 2 & 3):
```
Header  (16 bytes): magic=0x52435542  version=1  n_records(uint64)
Record  (21 bytes): corners[8] + edges[12] + cost[1]
Files split at 1 GB → dataset_000.bin, dataset_001.bin, ...
```

**State encoding** (Kociemba standard):
```
corners[c] = (position << 2) | orientation    pos∈[0,7], ori∈[0,2]
edges[e]   = (position << 1) | flip            pos∈[0,11], flip∈[0,1]
```

**Reading in Python** (Phase 2 / Phase 3):
```python
import numpy as np, struct

def read_dataset(path):
    with open(path, 'rb') as f:
        magic, version = struct.unpack('<II', f.read(8))
        n_records = struct.unpack('<Q', f.read(8))[0]
        raw = np.frombuffer(f.read(n_records * 21), dtype=np.uint8)
    records = raw.reshape(-1, 21)
    states = records[:, :20]   # (N, 20) — raw cubie array
    costs  = records[:, 20]    # (N,)    — optimal depth 0..20
    return states, costs
```

## Hardware

- CPU: Intel Core i7-11700F (8 cores / 16 threads)
- Storage: 800 GB on `D:\Search_DB\`
- OS: Windows 11, MinGW-w64 / GCC toolchain

## Status

| Phase | Status      | Notes                              |
|-------|-------------|------------------------------------|
| 1     | Complete    | All C++ code written, ready to build |
| 2     | Not started | Awaiting Phase 1 dataset           |
| 3     | Not started | Awaiting Phase 2 trained model     |
