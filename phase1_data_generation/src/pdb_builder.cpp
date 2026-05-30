// ---------------------------------------------------------------------------
// build_pdbs — one-shot generator for the additive 7-8 pattern databases.
//
// STATUS: skeleton.  The query path (AdditivePDB::rank/lookup/load/save) and
// the partition (puzzle_defs.h) are complete; this builder — the heavy
// retrograde BFS — is the next step.  generate_data currently runs on
// ManhattanHeuristic, so the pipeline is end-to-end without these tables.
//
// ALGORITHM (per group, e.g. A = {1..7}):
//   Retrograde, BLANK-AWARE 0-1 BFS from the goal arrangement.
//     * BFS node   = (positions of the group's k tiles) + (blank position).
//                    state space = 16 * 15 * ... * (16-k) entries.
//     * transition = slide the blank to an orthogonal neighbor cell:
//          - neighbor holds a GROUP tile      -> real pattern move, cost +1
//          - neighbor holds a don't-care tile -> cost +0
//       The mixed 0/1 edge weights make this a 0-1 BFS: push 0-cost successors
//       to the FRONT of a deque, 1-cost successors to the BACK.
//     * after BFS, PROJECT to the query table:
//          PDB[pattern_rank] = min over blank position of dist(pattern, blank)
//       (AdditivePDB::rank ignores the blank, so collapse over it here.)
//
// MEMORY (build time, uint8 distance array over the blank-aware space):
//   Group A (k=7): 16*..*9  =   518,918,400  (~0.5 GB)
//   Group B (k=8): 16*..*8  = 4,151,347,200  (~4.1 GB)
//   The target i7 box has the RAM; on a dev machine use a smaller partition
//   (edit group_of / sizes in puzzle_defs.h) to validate correctness.
//
// USAGE (planned):
//   build_pdbs --out <dir>     ->  writes pdb_a.bin, pdb_b.bin
// ---------------------------------------------------------------------------
#include "pdb.h"

#include <cstring>
#include <iostream>
#include <string>

using namespace puzzle;

static std::vector<int> group_tiles(int group_id) {
    std::vector<int> g;
    for (int t = 1; t < N_CELLS; ++t)
        if (group_of(t) == group_id) g.push_back(t);
    return g;
}

int main(int argc, char** argv) {
    std::string out_dir = ".";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_dir = argv[++i];
    }

    const auto a = group_tiles(0);
    const auto b = group_tiles(1);
    std::cerr << "build_pdbs: NOT YET IMPLEMENTED (skeleton).\n"
              << "  Group A (" << a.size() << " tiles) -> table " << pdb_size((int)a.size()) << " entries\n"
              << "  Group B (" << b.size() << " tiles) -> table " << pdb_size((int)b.size()) << " entries\n"
              << "  Output dir would be: " << out_dir << " (pdb_a.bin, pdb_b.bin)\n"
              << "  See the header comment in src/pdb_builder.cpp for the BFS algorithm.\n";
    return 1;
}
