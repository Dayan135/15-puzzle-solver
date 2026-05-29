# Phase 1 — Data Generation (C++)

## Goal

Generate 100M `[CubeState, OptimalCost]` pairs by solving uniformly random Rubik's Cube states optimally using IDA* with Pattern Database heuristics. Output is binary `.bin` files consumed by Phase 2.

## Architecture

Two executables:
- `build_pdbs` — BFS PDB generator. Run **once** before anything else (~2-3 hours).
- `generate_data` — Main pipeline (~2-4 days on i7-11700F with 14 threads).

```
Work Queue  ──►  14 Solver Threads (IDA*)  ──►  Result Queue  ──►  I/O Thread
    ▲                                                                    │
Main Thread                                                    dataset_NNN.bin
(state gen)                                                  (D:\Search_DB\)
```

## Toolchain

- **Compiler**: MinGW-w64 / GCC on Windows 11
- **Standard**: C++17
- **Threading**: `std::thread` + condition variables
- **PDB generation**: OpenMP (BFS parallelism)
- **Build system**: CMake with MinGW Makefiles

```bash
mkdir build && cd build
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
mingw32-make -j16
```

## Key Design Decisions

### State Representation (20 bytes)
```cpp
struct CubeState {
    uint8_t corners[8];  // (position << 2) | orientation  pos∈[0,7], ori∈[0,2]
    uint8_t edges[12];   // (position << 1) | flip          pos∈[0,11], flip∈[0,1]
};
```
Corner/edge index convention: **Kociemba standard**
- Corners: 0=URF 1=UFL 2=ULB 3=UBR 4=DFR 5=DLF 6=DBL 7=DRB
- Edges: 0=UR 1=UF 2=UL 3=UB 4=DR 5=DF 6=DL 7=DB 8=FR 9=FL 10=BL 11=BR

### Pattern Databases (RAM: ~86 MB total)
| PDB       | Size               | Tracks                      |
|-----------|--------------------|-----------------------------|
| corner    | 88,179,840 (44 MB) | all 8 corners               |
| edge_a    | 42,577,920 (21 MB) | UR,UF,UL,UB,FR,FL (0–3+8–9)|
| edge_b    | 42,577,920 (21 MB) | DR,DF,DL,DB,BL,BR (4–7+10–11)|

Heuristic = `max(corner_h, edge_a_h, edge_b_h)` — admissible.

PDB indices:
```
corner_index = perm_rank(corners) * 2187 + ori_rank(corners[0..6])
edge_X_index = k_perm_rank(edge_set_X) * 64 + flip_rank(edge_set_X)
```

### Disk Format (21 bytes/record → 2.1 GB for 100M)
```
File header (16 bytes): magic(4)=0x52435542 + version(4) + n_records(8)
Per record (21 bytes):  corners[8] + edges[12] + cost(1)
File split at 1 GB → dataset_000.bin, dataset_001.bin, ...
Output path: D:\Search_DB\
```

## Execution Order

```bash
# Step 1: Build PDBs (~2-3 hours, one-time)
./build/build_pdbs.exe --out D:/Search_DB/

# Step 2: Generate dataset (~2-4 days)
./build/generate_data.exe \
    --threads 14 \
    --target 100000000 \
    --pdb-dir D:/Search_DB/ \
    --out-dir D:/Search_DB/
```

## File Map

```
include/
  cube_defs.h       # Constants, enums, edge sets, PDB sizes
  move_tables.h     # MoveTables struct + inline apply_move()
  cube_state.h      # CubeState functions (solved, random, ranking)
  blocking_queue.h  # Thread-safe bounded queue
  pdb.h             # PatternDB + MaxHeuristic
  ida_star.h        # IDAStar solver class
  io_writer.h       # AsyncWriter + SolvedRecord
src/
  move_tables.cpp   # 6 quarter-turn tables → derives all 18 by composition
  cube_state.cpp    # Lehmer rank, k-perm rank, random walk
  pdb.cpp           # nibble-packed load/save, MaxHeuristic query
  pdb_builder.cpp   # BFS from goal (main for build_pdbs)
  ida_star.cpp      # IDA* with 3-level move pruning
  main.cpp          # Orchestrator (main for generate_data)
CMakeLists.txt
```

## Notes for Future Development

- **Verifying move tables**: Run `R U R' U'` six times from solved — must return to solved.
- **Resuming generation**: The generator doesn't checkpoint. If interrupted, existing `.bin` files remain valid; restart with a new seed and the same `--out-dir`.
- **Adding a heuristic**: To swap in a learned heuristic for Phase 3, replace `MaxHeuristic` in `IDAStar` with a templated heuristic parameter.
