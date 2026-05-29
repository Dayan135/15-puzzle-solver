#pragma once
#include "pdb.h"
#include "move_tables.h"

// IDA* solver. One instance per thread — all state is on the call stack.
class IDAStar {
public:
    IDAStar(const MoveTables& mt, const MaxHeuristic& h);

    // Returns the optimal cost (0–20) for `start`.
    uint8_t solve(const CubeState& start) const;

private:
    // Returns IDA_FOUND if a solution at depth <= bound is found,
    // otherwise returns the minimum f-value that exceeded bound.
    int search(const CubeState& s,
               uint8_t g, uint8_t bound,
               int last_move, int last_face) const;

    const MoveTables&  mt_;
    const MaxHeuristic& h_;
};
