// pdb_builder.cpp — standalone executable that generates all three PDB files.
// Usage: build_pdbs --out <dir>
//
// Generates: corner.pdb (~44 MB), edge_a.pdb (~21 MB), edge_b.pdb (~21 MB)
// Expected runtime on i7-11700F: ~2-3 hours total.

#include "pdb.h"
#include "cube_state.h"
#include "move_tables.h"
#include "cube_defs.h"

#include <iostream>
#include <vector>
#include <queue>
#include <string>
#include <chrono>
#include <cstring>
#include <filesystem>

// ---------------------------------------------------------------------------
// Corner PDB builder
// Tracks only corner positions and orientations (ignores edges).
// State key: (perm_rank * 2187 + ori_rank), stored as a single uint32_t index.
// ---------------------------------------------------------------------------
static void build_corner_pdb(const std::string& out_path) {
    using Clock = std::chrono::steady_clock;
    std::cout << "Building corner PDB (" << CORNER_PDB_SIZE << " entries)...\n" << std::flush;
    auto t0 = Clock::now();

    PatternDB pdb(CORNER_PDB_SIZE);
    const MoveTables& mt = MoveTables::get();

    CubeState goal = solved_state();
    uint32_t goal_idx = corner_perm_rank(goal) * 2187u + corner_ori_rank(goal);
    pdb.set(goal_idx, 0);

    // BFS over corner states only.
    // We represent each BFS node as a full CubeState, but only the corner
    // portion determines the index. Edges are kept at solved (canonical).
    struct Node { CubeState s; uint8_t depth; };
    std::queue<Node> bfs;
    bfs.push({goal, 0});

    uint64_t visited = 1;
    uint8_t  max_depth = 0;

    while (!bfs.empty()) {
        auto [s, d] = bfs.front(); bfs.pop();
        if (d > max_depth) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - t0).count();
            std::cout << "  depth " << static_cast<int>(d) << "  visited=" << visited
                      << "  elapsed=" << elapsed << "s\n" << std::flush;
            max_depth = d;
        }
        for (int m = 0; m < N_MOVES; ++m) {
            CubeState ns = apply_move(s, m, mt);
            uint32_t idx = corner_perm_rank(ns) * 2187u + corner_ori_rank(ns);
            if (pdb.get(idx) == 15u) { // unvisited
                pdb.set(idx, d + 1u);
                ++visited;
                if (visited == CORNER_PDB_SIZE) goto corner_done;
                bfs.push({ns, static_cast<uint8_t>(d + 1)});
            }
        }
    }
corner_done:
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - t0).count();
    std::cout << "  Done. " << visited << "/" << CORNER_PDB_SIZE
              << " entries in " << elapsed << "s\n" << std::flush;
    pdb.save(out_path);
    std::cout << "  Saved: " << out_path << "\n\n";
}

// ---------------------------------------------------------------------------
// 6-Edge PDB builder (generic, used for both edge_a and edge_b)
// Tracks only the 6 specified edge cubies; ignores corners and other edges.
// ---------------------------------------------------------------------------
static void build_edge_pdb(const int* edge_set, int k,
                            const std::string& label,
                            const std::string& out_path) {
    using Clock = std::chrono::steady_clock;
    std::cout << "Building " << label << " PDB (" << EDGE_PDB_SIZE << " entries)...\n" << std::flush;
    auto t0 = Clock::now();

    PatternDB pdb(EDGE_PDB_SIZE);
    const MoveTables& mt = MoveTables::get();

    CubeState goal = solved_state();
    uint32_t goal_idx = k_perm_rank(goal, edge_set, k) * 64u
                      + k_flip_rank(goal, edge_set, k);
    pdb.set(goal_idx, 0);

    struct Node { CubeState s; uint8_t depth; };
    std::queue<Node> bfs;
    bfs.push({goal, 0});

    uint64_t visited = 1;
    uint8_t  max_depth = 0;

    while (!bfs.empty()) {
        auto [s, d] = bfs.front(); bfs.pop();
        if (d > max_depth) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - t0).count();
            std::cout << "  depth " << static_cast<int>(d) << "  visited=" << visited
                      << "  elapsed=" << elapsed << "s\n" << std::flush;
            max_depth = d;
        }
        for (int m = 0; m < N_MOVES; ++m) {
            CubeState ns = apply_move(s, m, mt);
            uint32_t idx = k_perm_rank(ns, edge_set, k) * 64u
                         + k_flip_rank(ns, edge_set, k);
            if (pdb.get(idx) == 15u) {
                pdb.set(idx, d + 1u);
                ++visited;
                if (visited == EDGE_PDB_SIZE) goto edge_done;
                bfs.push({ns, static_cast<uint8_t>(d + 1)});
            }
        }
    }
edge_done:
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - t0).count();
    std::cout << "  Done. " << visited << "/" << EDGE_PDB_SIZE
              << " entries in " << elapsed << "s\n" << std::flush;
    pdb.save(out_path);
    std::cout << "  Saved: " << out_path << "\n\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string out_dir = ".";
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--out") out_dir = argv[i + 1];
    }

    // Ensure output directory exists
    std::filesystem::create_directories(out_dir);

    auto path = [&](const std::string& name) {
        return out_dir + "/" + name;
    };

    std::cout << "=== PDB Builder ===\n";
    std::cout << "Output dir: " << out_dir << "\n\n";

    // Initialize move tables once (triggers singleton construction)
    (void)MoveTables::get();

    build_corner_pdb(path("corner.pdb"));
    build_edge_pdb(EDGE_SET_A, 6, "edge_a", path("edge_a.pdb"));
    build_edge_pdb(EDGE_SET_B, 6, "edge_b", path("edge_b.pdb"));

    std::cout << "All PDBs built successfully.\n";
    return 0;
}
