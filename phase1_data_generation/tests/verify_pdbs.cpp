// verify_pdbs.cpp — sanity-checks all three PDB files.
// Usage: verify_pdbs --pdb-dir <dir>
//
// Checks performed:
//   1. Solved state: h(solved) == 0 for every PDB
//   2. Admissibility: for N-move scrambles, h(s) <= N  (for N = 1..20)
//   3. Consistency:   |h(s) - h(apply_move(s,m))| <= 1 for sampled states
//   4. Distribution:  prints histogram of h values over 10000 random states

#include "pdb.h"
#include "cube_state.h"
#include "move_tables.h"
#include "cube_defs.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <cstdlib>

// ---- helpers ---------------------------------------------------------------

static std::string pdb_path(const std::string& dir, const std::string& name) {
    return dir + "/" + name;
}

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        std::exit(1);
    }
}

// Generate a state by applying exactly n random moves (no immediate reversal)
static CubeState scramble(int n, std::mt19937& rng) {
    const MoveTables& mt = MoveTables::get();
    CubeState s = solved_state();
    int last_face = -1;
    for (int i = 0; i < n; ++i) {
        int m;
        do { m = static_cast<int>(rng() % N_MOVES); }
        while (last_face >= 0 && MOVE_FACE[m] == last_face);
        s = apply_move(s, m, mt);
        last_face = MOVE_FACE[m];
    }
    return s;
}

// ---- test 1: solved state --------------------------------------------------

static void test_solved(const MaxHeuristic& h) {
    std::cout << "Test 1 — Solved state h == 0 ... ";
    CubeState sol = solved_state();
    uint8_t val = h(sol);
    check(val == 0, "h(solved) = " + std::to_string(val) + ", expected 0");
    std::cout << "PASS (h=" << static_cast<int>(val) << ")\n";
}

// ---- test 2: admissibility -------------------------------------------------

static void test_admissibility(const MaxHeuristic& h) {
    std::cout << "Test 2 — Admissibility h(s) <= moves_applied ...\n";
    std::mt19937 rng(42);
    int failures = 0;
    for (int depth = 1; depth <= 20; ++depth) {
        int local_fail = 0;
        for (int trial = 0; trial < 200; ++trial) {
            CubeState s = scramble(depth, rng);
            uint8_t hval = h(s);
            if (hval > static_cast<uint8_t>(depth)) {
                ++local_fail;
                ++failures;
            }
        }
        std::cout << "  depth " << std::setw(2) << depth
                  << ": " << (local_fail == 0 ? "PASS" : "FAIL")
                  << " (" << local_fail << " violations)\n";
    }
    check(failures == 0, std::to_string(failures) + " admissibility violations");
    std::cout << "  All admissibility checks PASSED.\n";
}

// ---- test 3: consistency ---------------------------------------------------

static void test_consistency(const MaxHeuristic& h) {
    std::cout << "Test 3 — Consistency |h(s) - h(next)| <= 1 ... ";
    const MoveTables& mt = MoveTables::get();
    std::mt19937 rng(99);
    int failures = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        CubeState s = scramble(25, rng);
        uint8_t hs = h(s);
        for (int m = 0; m < N_MOVES; ++m) {
            CubeState ns = apply_move(s, m, mt);
            uint8_t hns = h(ns);
            int diff = static_cast<int>(hs) - static_cast<int>(hns);
            if (diff < -1 || diff > 1) {
                ++failures;
                if (failures <= 3)
                    std::cerr << "\n  INCONSISTENT: h(s)=" << static_cast<int>(hs)
                              << " h(next)=" << static_cast<int>(hns)
                              << " move=" << m;
            }
        }
    }
    check(failures == 0, std::to_string(failures) + " consistency violations");
    std::cout << "PASS\n";
}

// ---- test 4: distribution --------------------------------------------------

static void test_distribution(const MaxHeuristic& h) {
    std::cout << "Test 4 — h-value distribution over 10000 random states:\n";
    std::mt19937 rng(7);
    uint64_t hist[21] = {};
    for (int i = 0; i < 10000; ++i) {
        uint8_t hval = h(scramble(100, rng));
        if (hval <= 20) ++hist[hval];
    }
    for (int d = 0; d <= 20; ++d) {
        if (hist[d] == 0) continue;
        std::cout << "  h=" << std::setw(2) << d
                  << "  " << std::setw(5) << hist[d]
                  << " (" << std::fixed << std::setprecision(1)
                  << hist[d] / 100.0 << "%)\n";
    }
    // Sanity checks on the distribution:
    // - h=0 should be extremely rare (only the solved state itself)
    // - >95% of random states should have h >= 7
    //   (corner PDB max is ~10, edge PDB max is ~9; random states cluster at 8-10)
    check(hist[0] == 0, "h=0 seen for a random state — move tables may be broken");
    uint64_t high = 0;
    for (int d = 7; d <= 20; ++d) high += hist[d];
    check(high >= 9500, "Expected >95% of states with h>=7, got " +
          std::to_string(high / 100) + "%");
    std::cout << "  Distribution looks correct.\n";
}

// ---- main ------------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::string pdb_dir = ".";
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--pdb-dir") pdb_dir = argv[i + 1];

    std::cout << "=== PDB Verification ===\n"
              << "PDB dir: " << pdb_dir << "\n\n";

    (void)MoveTables::get();

    std::cout << "Loading PDBs...\n";
    PatternDB corner_pdb(CORNER_PDB_SIZE);
    PatternDB edge_a_pdb(EDGE_PDB_SIZE);
    PatternDB edge_b_pdb(EDGE_PDB_SIZE);
    corner_pdb.load(pdb_path(pdb_dir, "corner.pdb"));
    edge_a_pdb.load(pdb_path(pdb_dir, "edge_a.pdb"));
    edge_b_pdb.load(pdb_path(pdb_dir, "edge_b.pdb"));
    std::cout << "Loaded.\n\n";

    MaxHeuristic h(corner_pdb, edge_a_pdb, edge_b_pdb);

    test_solved(h);
    test_admissibility(h);
    test_consistency(h);
    test_distribution(h);

    std::cout << "\n=== All tests PASSED ===\n";
    return 0;
}
