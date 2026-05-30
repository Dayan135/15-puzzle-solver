#pragma once
#include "puzzle_defs.h"
#include <array>
#include <cstdint>
#include <random>

// ---------------------------------------------------------------------------
// State = one nibble-packed 15-puzzle board in a single uint64_t.
//
// All transitions are O(1) bitwise ops.  The blank position is NOT stored in
// the word; it is found once at the search root (find_blank) and then carried
// incrementally as the search slides tiles, so the hot loop never rescans.
// ---------------------------------------------------------------------------
namespace puzzle {

using State = uint64_t;

// tile value at cell `pos` (0..15; 0 = blank).
inline int tile_at(State s, int pos) {
    return static_cast<int>((s >> (pos * BITS)) & 0xFULL);
}

// board with cell `pos` set to `tile` (other cells untouched).
inline State set_tile(State s, int pos, int tile) {
    return (s & ~(0xFULL << (pos * BITS))) | (static_cast<uint64_t>(tile) << (pos * BITS));
}

// linear scan for the blank; used once per solve at the root.
inline int find_blank(State s) {
    for (int p = 0; p < N_CELLS; ++p)
        if (((s >> (p * BITS)) & 0xFULL) == 0) return p;
    return -1; // never happens for a valid board
}

// Slide the tile at neighbor cell `n` into the blank at `blank`.
// Caller guarantees `n` is orthogonally adjacent to `blank`.
inline State slide(State s, int blank, int n) {
    int t = tile_at(s, n);
    s = set_tile(s, blank, t);
    s = set_tile(s, n, 0);
    return s;
}

// Orthogonal neighbors of each cell (-1 padded), and how many each has (2..4).
extern const std::array<std::array<int8_t, 4>, N_CELLS> NEIGHBORS;
extern const std::array<int8_t, N_CELLS>                NEIGHBOR_COUNT;

// Apply `n_moves` random VALID slides from `start`, never immediately reversing
// the previous move.  Move-based scrambling guarantees the result is solvable
// (it is reachable from `start`); a random nibble permutation would be
// unsolvable ~50% of the time.
State scramble(State start, int n_moves, std::mt19937_64& rng);

// True iff `s` is reachable from GOAL (full-permutation parity vs. blank
// taxicab-distance parity).  Used by tests; generated states are solvable by
// construction.
bool is_solvable(State s);

// Sum of Manhattan distances of all non-blank tiles.  Admissible; used as the
// placeholder heuristic until the additive PDBs are built.
int manhattan(State s);

} // namespace puzzle
