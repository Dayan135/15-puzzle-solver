#include "ida_star.h"
#include "cube_state.h"
#include <algorithm>

IDAStar::IDAStar(const MoveTables& mt, const MaxHeuristic& h)
    : mt_(mt), h_(h) {}

uint8_t IDAStar::solve(const CubeState& start) const {
    // Trivial case: already solved (depth 0)
    if (is_solved(start)) return 0;

    uint8_t bound = h_(start);
    while (true) {
        int t = search(start, 0, bound, NO_MOVE, -1);
        if (t == IDA_FOUND) return bound;
        // t is the minimum f that exceeded the previous bound → next threshold
        bound = static_cast<uint8_t>(t);
    }
}

int IDAStar::search(const CubeState& s,
                    uint8_t g, uint8_t bound,
                    int last_move, int last_face) const {
    const uint8_t h_val = h_(s);
    const int     f     = static_cast<int>(g) + static_cast<int>(h_val);

    if (f > static_cast<int>(bound)) return f;

    // Check goal: h==0 is necessary but not sufficient (corner + 2 edge PDBs
    // can simultaneously be 0 only at the solved state for a valid cube).
    if (h_val == 0 && is_solved(s)) return IDA_FOUND;

    int min_t = IDA_INF;

    for (int m = 0; m < N_MOVES; ++m) {
        // Pruning 1: never immediately invert the last move
        if (last_move != NO_MOVE && MOVE_INVERSE[m] == last_move) continue;

        const int face = MOVE_FACE[m];

        // Pruning 2: don't apply two moves on the same face in a row
        // (e.g. U then U or U then U2 — equivalent to a single move)
        if (face == last_face) continue;

        // Pruning 3: for commuting opposite-face pairs (U/D, L/R, F/B),
        // enforce a canonical order so we don't generate both U D and D U.
        // Allow only face < opposite when last was the opposite face.
        if (last_face >= 0 && OPPOSITE_FACE[face] == last_face && face > last_face) continue;

        CubeState ns = apply_move(s, m, mt_);
        int t = search(ns, g + 1u, bound, m, face);
        if (t == IDA_FOUND) return IDA_FOUND;
        if (t < min_t)      min_t = t;
    }
    return min_t;
}
