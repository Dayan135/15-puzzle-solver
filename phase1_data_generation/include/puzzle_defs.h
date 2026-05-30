#pragma once
#include <cstdint>

// ---------------------------------------------------------------------------
// 15-Puzzle board geometry and the additive-PDB tile partition.
//
// The board is a 4x4 grid of 16 cells (indices 0..15, row-major).  Cell `p`
// holds a tile value in 0..15, where 0 is the blank.  The whole board packs
// into a single uint64_t: 16 nibbles, nibble `p` = tile at cell `p`.
// ---------------------------------------------------------------------------
namespace puzzle {

constexpr int      N_CELLS = 16;  // 4x4
constexpr int      SIDE    = 4;
constexpr int      BITS    = 4;   // bits per tile

// Goal: cell p holds tile p; blank (tile 0) sits at cell 0.
//   nibble p == p  ->  0xFEDCBA9876543210
constexpr uint64_t GOAL = 0xFEDCBA9876543210ULL;

constexpr int row_of(int cell) { return cell / SIDE; }
constexpr int col_of(int cell) { return cell % SIDE; }

// ---------------------------------------------------------------------------
// Additive Pattern Database partition (7-8 split), blank (tile 0) excluded.
//   Group A = {1,2,3,4,5,6,7}   (7 tiles)
//   Group B = {8,9,10,11,12,13,14,15} (8 tiles)
// Because the groups are disjoint and every slide moves exactly one tile, the
// two PDB costs count disjoint move sets and may be SUMMED while staying
// admissible (cf. ManhattanHeuristic / SumHeuristic in pdb.h).
// ---------------------------------------------------------------------------
constexpr int GROUP_A_SIZE = 7;
constexpr int GROUP_B_SIZE = 8;

// group_of(tile): 0 = group A, 1 = group B, -1 = blank.
constexpr int group_of(int tile) {
    if (tile == 0) return -1;
    return (tile <= GROUP_A_SIZE) ? 0 : 1;
}

// Dense PDB table size for a group of k tiles = 16 * 15 * ... * (16-k+1).
constexpr uint64_t pdb_size(int k) {
    uint64_t n = 1;
    for (int i = 0; i < k; ++i) n *= static_cast<uint64_t>(N_CELLS - i);
    return n;
}

} // namespace puzzle
