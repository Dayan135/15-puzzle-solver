#pragma once
#include "move_tables.h" // also includes cube_defs.h; CubeState defined there
#include <random>
#include <cstdint>

// Solved state: cubie c at position c, orientation/flip = 0
CubeState solved_state() noexcept;

// True iff every cubie is in its home position with zero twist/flip
bool is_solved(const CubeState& s) noexcept;

// Random state via 100-move uniform random walk.
// Avoids immediately reversing the previous move (stays uniform).
// mt19937 is pseudo-random number generator
CubeState random_state(std::mt19937& rng) noexcept;

// ---------------------------------------------------------------------------
// Ranking functions used by PDB indexing
// ---------------------------------------------------------------------------

// Lehmer code rank of the corner permutation → [0, 8!-1 = 40319]
uint32_t corner_perm_rank(const CubeState& s) noexcept;

// Base-3 rank of corner orientations (first 7 corners; 8th is determined)
// Result in [0, 3^7-1 = 2186]
uint32_t corner_ori_rank(const CubeState& s) noexcept;

// Rank of the k-permutation formed by the positions of the k edges listed in
// edge_set[], treating them as an ordered partial permutation out of 12 slots.
// Result in [0, P(12,k)-1].
uint32_t k_perm_rank(const CubeState& s, const int* edge_set, int k) noexcept;

// Bit-packed flip rank of the k edges listed in edge_set[].
// Bit i = flip of edge_set[i].  Result in [0, 2^k-1].
uint8_t k_flip_rank(const CubeState& s, const int* edge_set, int k) noexcept;
