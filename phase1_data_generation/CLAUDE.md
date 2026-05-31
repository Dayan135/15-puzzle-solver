# Phase 1 — Data Generation (C++)

## Goal

Generate a massive dataset of `[State, OptimalCost]` pairs by solving uniformly random 15-puzzle
states optimally using IDA* + admissible heuristics. Output is compact binary `.bin` files consumed
by Phase 2 (neural network training).

## Architecture

Two executables:
- `build_pdbs` — one-shot additive PDB generator (retrograde blank-aware 0-1 BFS).
- `generate_data` — main pipeline; runs IDA* with the additive 7-8 `SumHeuristic`, generating stratified `[state, cost]` pairs and reporting per-bucket solve time and mean cost.

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
| Additive 7-8 PDBs | ✅ In use        | `SumHeuristic` over disjoint groups {1..7}+{8..15}; production heuristic. |
| Manhattan         | ✅ Available     | `ManhattanHeuristic` kept as a no-setup fallback; not used by `generate_data`. |

`IDAStar` is templated on `H`, so the heuristic is a one-line swap
(`IDAStar<SumHeuristic>` in `main.cpp`). The PDBs are built once by
`build_pdbs` and loaded read-only, shared across all worker threads.

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

### Additive PDBs — 7-8 split
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

# build the additive 7-8 PDBs once (~30 min on the cluster, ~4 GB RAM for group B)
./build/build_pdbs --out ./data/pdbs

# generate a stratified dataset (equal quota across scramble-length buckets)
./build/generate_data \
    --threads 16 \
    --target 100000000 \
    --pdb-dir ./data/pdbs \
    --out-dir ./data/full \
    --seed 42
# optional: --buckets "1,5,10,30,100,1000" to override the default 22 buckets
```

On the cluster, submit via the job scripts in `jobs/` (`puzzle_pdbs_*.sh` then
`puzzle_full_*.sh`); both skip the PDB build if `data/pdbs/` is already populated.

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
