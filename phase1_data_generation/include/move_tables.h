#pragma once
#include "cube_defs.h"

// ---------------------------------------------------------------------------
// CubeState — compact cubie-centric representation
// Declared here (ahead of MoveTables) so apply_move can be inlined.
//
// corners[c] = (position << 2) | orientation
//   position    in [0, 7],  which of the 8 corner slots the cubie occupies
//   orientation in [0, 2],  Kociemba twist (0 = UD sticker faces U/D)
//
// edges[e] = (position << 1) | flip
//   position in [0, 11], which of the 12 edge slots the cubie occupies
//   flip     in [0, 1],  Kociemba flip (0 = correctly oriented)
// ---------------------------------------------------------------------------
struct CubeState {
    uint8_t corners[N_CORNERS]; // 8 bytes
    uint8_t edges[N_EDGES];     // 12 bytes
};                              // 20 bytes total

// ---------------------------------------------------------------------------
// MoveTables — precomputed lookup tables for all 18 moves
//
// corner_pos[m][old_pos] → new position of a cubie that was at old_pos
// corner_ori[m][old_pos] → orientation addend (mod 3) for that cubie
// edge_pos  [m][old_pos] → new position of an edge cubie
// edge_flip [m][old_pos] → flip XOR for that edge cubie
// ---------------------------------------------------------------------------
struct MoveTables {
    uint8_t corner_pos[N_MOVES][N_CORNERS];
    uint8_t corner_ori[N_MOVES][N_CORNERS];
    uint8_t edge_pos  [N_MOVES][N_EDGES];
    uint8_t edge_flip [N_MOVES][N_EDGES];

    // Singleton — constructed once from hardcoded quarter-turn data
    static const MoveTables& get();

private:
    MoveTables();
    static void compose_corner_pos(const uint8_t a[8], const uint8_t b[8], uint8_t out[8]);
    static void compose_corner_ori(const uint8_t pa[8], const uint8_t oa[8],
                                   const uint8_t pb[8], const uint8_t ob[8],
                                   uint8_t out_pos[8], uint8_t out_ori[8]);
    static void compose_edge_pos  (const uint8_t a[12], const uint8_t b[12], uint8_t out[12]);
    static void compose_edge_full (const uint8_t pa[12], const uint8_t fa[12],
                                   const uint8_t pb[12], const uint8_t fb[12],
                                   uint8_t out_pos[12], uint8_t out_flip[12]);
};

// ---------------------------------------------------------------------------
// apply_move — inlined hot-path function for IDA* tree expansion
// ---------------------------------------------------------------------------
inline CubeState apply_move(const CubeState& s, int m, const MoveTables& mt) noexcept {
    CubeState r;
    for (int c = 0; c < N_CORNERS; ++c) {
        const uint8_t op = s.corners[c] >> 2;
        const uint8_t oo = s.corners[c] & 3u;
        const uint8_t np = mt.corner_pos[m][op];
        const uint8_t no = static_cast<uint8_t>((oo + mt.corner_ori[m][op]) % 3u);
        r.corners[c] = static_cast<uint8_t>((np << 2) | no);
    }
    for (int e = 0; e < N_EDGES; ++e) {
        const uint8_t op = s.edges[e] >> 1;
        const uint8_t of = s.edges[e] & 1u;
        const uint8_t np = mt.edge_pos[m][op];
        const uint8_t nf = static_cast<uint8_t>(of ^ mt.edge_flip[m][op]);
        r.edges[e] = static_cast<uint8_t>((np << 1) | nf);
    }
    return r;
}
