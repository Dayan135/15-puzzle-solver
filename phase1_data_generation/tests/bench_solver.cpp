// bench_solver.cpp — measures IDA* solve time for scrambles at depths 5, 10, 15
// Usage: bench_solver --pdb-dir <dir>

#include "pdb.h"
#include "cube_state.h"
#include "move_tables.h"
#include "ida_star.h"
#include "cube_defs.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <random>

static std::string pdb_path(const std::string& dir, const std::string& name) {
    return dir + "/" + name;
}

static CubeState exact_scramble(int depth, std::mt19937& rng) {
    const MoveTables& mt = MoveTables::get();
    CubeState s = solved_state();
    int last_face = -1;
    for (int i = 0; i < depth; ++i) {
        int m;
        do { m = static_cast<int>(rng() % N_MOVES); }
        while (last_face >= 0 && MOVE_FACE[m] == last_face);
        s = apply_move(s, m, mt);
        last_face = MOVE_FACE[m];
    }
    return s;
}

int main(int argc, char* argv[]) {
    std::string pdb_dir = ".";
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--pdb-dir") pdb_dir = argv[i + 1];

    (void)MoveTables::get();

    PatternDB corner_pdb(CORNER_PDB_SIZE);
    PatternDB edge_a_pdb(EDGE_PDB_SIZE);
    PatternDB edge_b_pdb(EDGE_PDB_SIZE);
    corner_pdb.load(pdb_path(pdb_dir, "corner.pdb"));
    edge_a_pdb.load(pdb_path(pdb_dir, "edge_a.pdb"));
    edge_b_pdb.load(pdb_path(pdb_dir, "edge_b.pdb"));

    MaxHeuristic h(corner_pdb, edge_a_pdb, edge_b_pdb);
    IDAStar solver(MoveTables::get(), h);

    std::mt19937 rng(123);

    // Test depths 5, 10, 14 — deep states would take too long to include here
    for (int depth : {5, 10, 14}) {
        std::cout << "--- Scramble depth " << depth << " ---\n";
        const int N = (depth <= 10) ? 20 : 5;
        double total_ms = 0;
        for (int trial = 0; trial < N; ++trial) {
            CubeState s = exact_scramble(depth, rng);
            uint8_t hval = h(s);
            auto t0 = std::chrono::high_resolution_clock::now();
            uint8_t cost = solver.solve(s);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_ms += ms;
            std::cout << "  trial " << std::setw(2) << trial
                      << "  h=" << std::setw(2) << static_cast<int>(hval)
                      << "  cost=" << std::setw(2) << static_cast<int>(cost)
                      << "  " << std::fixed << std::setprecision(1) << ms << "ms\n";
        }
        std::cout << "  avg: " << std::fixed << std::setprecision(1)
                  << total_ms / N << "ms per state\n\n";
    }
    return 0;
}
