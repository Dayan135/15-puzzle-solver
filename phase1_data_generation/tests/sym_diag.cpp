// sym_diag.cpp — checks that conjugating a single-move state by every symmetry
// yields another single-move state (the defining property of cube symmetries).

#include "symmetry.h"
#include "move_tables.h"
#include "cube_state.h"
#include "cube_defs.h"
#include <iostream>
#include <cstring>

static bool eq(const CubeState& a, const CubeState& b) {
    return std::memcmp(a.corners, b.corners, 8) == 0
        && std::memcmp(a.edges, b.edges, 12) == 0;
}

int main() {
    const MoveTables& mt = MoveTables::get();
    const SymmetryTables& st = SymmetryTables::get();

    // Precompute the 18 single-move states
    CubeState single[N_MOVES];
    for (int m = 0; m < N_MOVES; ++m)
        single[m] = apply_move(solved_state(), m, mt);

    int total_bad = 0;
    for (int m = 0; m < N_MOVES; ++m) {
        int bad = 0;
        for (int i = 0; i < SymmetryTables::N_SYM; ++i) {
            CubeState conj = apply_symmetry(single[m], i, st);
            // conj must equal one of the 18 single-move states
            bool ok = false;
            for (int m2 = 0; m2 < N_MOVES; ++m2)
                if (eq(conj, single[m2])) { ok = true; break; }
            if (!ok) ++bad;
        }
        if (bad) {
            std::cout << "move " << m << ": " << bad
                      << "/48 conjugates are NOT single moves\n";
            total_bad += bad;
        }
    }
    if (total_bad == 0)
        std::cout << "ALL conjugates of all single moves are single moves — PASS\n";
    else
        std::cout << "TOTAL bad conjugates: " << total_bad << "\n";

    // Detailed dump for move R (index 9) under first few symmetries
    std::cout << "\nDetail — conjugates of R (move 9):\n";
    for (int i = 0; i < 8; ++i) {
        CubeState conj = apply_symmetry(single[9], i, st);
        int which = -1;
        for (int m2 = 0; m2 < N_MOVES; ++m2) if (eq(conj, single[m2])) { which = m2; break; }
        std::cout << "  sym " << i << " -> "
                  << (which >= 0 ? ("move " + std::to_string(which)) : "NOT a single move")
                  << "\n";
    }
    return 0;
}
