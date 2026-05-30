#pragma once
#include "puzzle_defs.h"
#include "puzzle_state.h"
#include <limits>

// ---------------------------------------------------------------------------
// Iterative-Deepening A*.  Templated on the heuristic so the same code runs on
// ManhattanHeuristic now and SumHeuristic (additive PDBs) later with zero
// virtual-call overhead in the hot DFS loop.
//
// solve() returns the OPTIMAL cost-to-go (the label written to the dataset).
// The heuristic must be admissible for the result to be optimal.
//
// Each worker thread owns its own IDAStar; the heuristic is held by const
// reference and must be read-only / thread-safe (the PDB tables are).
// ---------------------------------------------------------------------------
namespace puzzle {

template <class H>
class IDAStar {
public:
    explicit IDAStar(const H& h) : h_(h) {}

    int solve(State start) {
        const int blank = find_blank(start);
        int bound = h_.h(start, blank);
        for (;;) {
            int next = std::numeric_limits<int>::max();
            const int found = dfs(start, blank, 0, bound, -1, next);
            if (found >= 0) return found;                       // goal reached
            if (next == std::numeric_limits<int>::max()) return -1; // unsolvable
            bound = next;
        }
    }

private:
    // Depth-first within f <= bound.  Returns solution depth if found, else -1;
    // raises `next` to the smallest f that exceeded the bound.
    int dfs(State s, int blank, int g, int bound, int prev, int& next) {
        const int f = g + h_.h(s, blank);
        if (f > bound) {
            if (f < next) next = f;
            return -1;
        }
        if (s == GOAL) return g;

        const int cnt = NEIGHBOR_COUNT[blank];
        for (int i = 0; i < cnt; ++i) {
            const int n = NEIGHBORS[blank][i];
            if (n == prev) continue;                  // never undo the last slide
            const int r = dfs(slide(s, blank, n), n, g + 1, bound, blank, next);
            if (r >= 0) return r;
        }
        return -1;
    }

    const H& h_;
};

} // namespace puzzle
