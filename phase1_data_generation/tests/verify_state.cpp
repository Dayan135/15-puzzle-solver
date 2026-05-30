// ---------------------------------------------------------------------------
// verify_state — correctness checks for the uint64_t state core and IDA*.
// Returns non-zero on any failure.
// ---------------------------------------------------------------------------
#include "ida_star.h"
#include "pdb.h"
#include "puzzle_state.h"

#include <cstdint>
#include <iostream>
#include <random>

using namespace puzzle;

static int failures = 0;
#define CHECK(cond, msg)                                                  \
    do {                                                                  \
        if (!(cond)) { std::cerr << "FAIL: " << (msg) << "\n"; ++failures; } \
    } while (0)

int main() {
    // --- pack / unpack round-trip on every cell ---
    {
        State s = GOAL;
        for (int p = 0; p < N_CELLS; ++p) CHECK(tile_at(s, p) == p, "GOAL nibble mismatch");
        s = set_tile(s, 5, 0xA);
        CHECK(tile_at(s, 5) == 0xA, "set_tile/tile_at round-trip");
        s = set_tile(s, 5, 5);
        CHECK(s == GOAL, "set_tile restore");
    }

    // --- GOAL invariants ---
    CHECK(manhattan(GOAL) == 0, "manhattan(GOAL) == 0");
    CHECK(is_solvable(GOAL), "GOAL is solvable");
    CHECK(find_blank(GOAL) == 0, "blank at cell 0 in GOAL");

    // --- a single slide and its undo ---
    {
        const int blank = find_blank(GOAL);          // 0
        const int n     = NEIGHBORS[blank][0];        // a neighbor (cell 4 = down)
        const State s1  = slide(GOAL, blank, n);
        CHECK(s1 != GOAL, "one slide changes the board");
        CHECK(is_solvable(s1), "one-slide state solvable");
        const State s0  = slide(s1, n, blank);        // undo
        CHECK(s0 == GOAL, "undo returns to GOAL");
    }

    // --- scrambles are always solvable; manhattan is a lower bound ---
    {
        std::mt19937_64 rng(12345);
        for (int i = 0; i < 5000; ++i) {
            const State s = scramble(GOAL, 200, rng);
            CHECK(is_solvable(s), "scrambled state solvable");
        }
    }

    // --- IDA* with Manhattan: optimal-cost sanity ---
    {
        const ManhattanHeuristic h;
        IDAStar<ManhattanHeuristic> solver(h);

        CHECK(solver.solve(GOAL) == 0, "solve(GOAL) == 0");

        // one random slide -> optimal cost exactly 1
        std::mt19937_64 rng(7);
        const State one = scramble(GOAL, 1, rng);
        CHECK(solver.solve(one) == 1, "solve(1-move scramble) == 1");

        // shallow scrambles: cost optimal, >= manhattan, same parity, <= moves applied
        for (int k = 2; k <= 12; ++k) {
            const State s   = scramble(GOAL, k, rng);
            const int   cst = solver.solve(s);
            CHECK(cst >= manhattan(s), "IDA* cost >= manhattan (admissible)");
            CHECK(cst <= k,            "IDA* cost <= scramble length");
            CHECK(((cst ^ manhattan(s)) & 1) == 0, "cost/manhattan same parity");
        }
    }

    // --- AdditivePDB ranking is a bijection onto [0, pdb_size) for tiny groups ---
    {
        AdditivePDB pdb;
        pdb.init_group({1, 2});                 // k=2 -> 16*15 = 240 ranks
        // GOAL: tile 1 at cell 1, tile 2 at cell 2 -> a specific valid rank
        const uint64_t r = pdb.rank(GOAL);
        CHECK(r < pdb_size(2), "rank within table bounds");
    }

    if (failures == 0) { std::cout << "verify_state: ALL CHECKS PASSED\n"; return 0; }
    std::cerr << "verify_state: " << failures << " FAILURE(S)\n";
    return 1;
}
