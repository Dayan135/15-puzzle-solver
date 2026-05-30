# Phase 1 — Data Generation (C++)

## Goal

Generate a massive dataset of `[State, OptimalCost]` pairs by solving uniformly random 15-puzzle
states optimally using IDA* + admissible heuristics. Output is compact binary `.bin` files consumed
by Phase 2 (neural network training).

## Architecture

Two executables:
- `build_pdbs` — one-shot additive PDB generator (skeleton; see next step below).
- `generate_data` — main pipeline; currently runs on the Manhattan heuristic placeholder.

```
main thread (generator)
  scramble(GOAL, 1000 moves) ──► BlockingQueue<State>
                                       │
                              N worker threads (IDA*)
                                       │
                              BlockingQueue<SolvedRecord>
                                       │
                               I/O thread (AsyncWriter)
                                       │
                               dataset_NNN.bin
```

## Heuristic status

| Heuristic         | Status          | Notes                                              |
|-------------------|-----------------|----------------------------------------------------|
| Manhattan         | ✅ Working       | Placeholder; weak on deep states, but **correct**  |
| Additive 7-8 PDBs | 🔲 Next step     | Builder skeleton in `pdb_builder.cpp`; swap is one line in `main.cpp` |

Swapping heuristics is a **one-line change** — `IDAStar` is templated on `H`:
```cpp
// Now:   IDAStar<ManhattanHeuristic> solver(heur);
// After: IDAStar<SumHeuristic>       solver(heur);
```

## Toolchain

- **Standard**: C++17
- **Build**: CMake 3.20+ with `cmake -S . -B build && cmake --build build -j`
- **Dev build** (no cmake): `clang++ -std=c++17 -O2 -pthread -Iinclude src/*.cpp src/main.cpp -o generate_data`
- **Threading**: `std::thread` + `BlockingQueue` (condition variables)
- **OpenMP**: optional; reserved for the future PDB BFS builder

## Key Design Decisions

### State — single `uint64_t`
Nibble at position `p` holds the tile at cell `p`. Goal = `0xFEDCBA9876543210`
(tile `p` at cell `p`; blank tile 0 at cell 0).
```
O(1) ops:  tile_at, set_tile, slide
```
The blank position is found once at the root (`find_blank`) and carried incrementally through the
DFS — the hot loop never rescans.

### Scrambling — move-based (solvability guaranteed)
`scramble(GOAL, n_moves, rng)` applies valid random slides, never immediately reversing the prior
move. **Never use a random nibble permutation** — ~50% of those are unsolvable.

### Additive PDBs — 7-8 split (next step)
Group A = {1..7}, Group B = {8..15}. Because the groups are **disjoint** and each slide moves
exactly one tile, the two PDB costs cover disjoint move sets and can be **summed** (admissible).
This is fundamentally different from Rubik's PDBs, where overlapping cubies force a `max`.

Memory (build-time): Group A ≈ 0.5 GB, Group B ≈ 4.1 GB. Target i7 box has the RAM.

### File format (9 bytes/record)
```
Header (16 bytes): magic=0x50313544 ("P15D") + version=1 + n_records (uint64, patched on close)
Record  (9 bytes): uint64 state (LE) + uint8 cost
Files roll at 1 GB → dataset_000.bin, dataset_001.bin, ...
```

Reading in Python (Phase 2):
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

## Build & Run

```bash
# configure + build all targets
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# correctness test (always run first)
./build/verify_state

# (future) build PDBs once, ~hours, needs ~4 GB RAM for group B
./build/build_pdbs --out /path/to/db/

# generate dataset
./build/generate_data \
    --threads 14 \
    --target 10000000 \
    --scramble 1000 \
    --out-dir /path/to/output/ \
    --seed 42
```

## File Map

```
include/
  puzzle_defs.h      # N_CELLS, GOAL, neighbor table, tile→group map, pdb_size()
  puzzle_state.h     # uint64_t State, tile_at, slide, find_blank, scramble, manhattan
  blocking_queue.h   # thread-safe bounded queue (generic, reused from prior project)
  pdb.h              # AdditivePDB (rank/load/save), SumHeuristic, ManhattanHeuristic
  ida_star.h         # IDAStar<H> — templated IDA* with reverse-move pruning
  io_writer.h        # AsyncWriter, SolvedRecord (9-byte binary format)
src/
  puzzle_state.cpp   # neighbor-table init, scramble, is_solvable, manhattan
  pdb.cpp            # partial-perm rank, nibble load/save
  pdb_builder.cpp    # 0-1 retrograde BFS skeleton (next step)
  main.cpp           # orchestrator for generate_data
tests/
  verify_state.cpp   # pack/slide round-trip, solvability invariant, IDA* optimality spot checks
CMakeLists.txt
```

## Commit scopes

- `gen` — state generation, scrambling, orchestrator
- `pdb` — pattern database builder and query path
- `ida` — IDA* solver
- `io` — binary file format and async writer
- `tests` — correctness tests
- `build` — CMakeLists.txt
- `docs` — CLAUDE.md files
