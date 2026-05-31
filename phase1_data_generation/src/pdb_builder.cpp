// ---------------------------------------------------------------------------
// build_pdbs — one-shot generator for the additive 7-8 pattern databases.
//
// ALGORITHM (per group, e.g. A = {1..7}):
//   Retrograde, BLANK-AWARE 0-1 BFS from the goal arrangement.
//     * BFS node   = (positions of the group's k tiles) + (blank position),
//                    ranked as a (k+1)-partial permutation.
//                    state space = 16 * 15 * ... * (16-k) entries.
//     * transition = slide the blank to an orthogonal neighbor cell:
//          - neighbor holds a GROUP tile      -> real pattern move, cost +1
//          - neighbor holds a don't-care tile -> cost +0
//       The mixed 0/1 edge weights make this a 0-1 BFS: push 0-cost successors
//       to the FRONT of a deque, 1-cost successors to the BACK.
//     * PROJECT to the query table on every distance improvement:
//          PDB[pattern_rank] = min over blank position of dist(pattern, blank)
//       (AdditivePDB::rank ignores the blank, so collapse over it here.)
//
//   NOTE: we use the *relaxation* form of 0-1 BFS (set distance on improvement,
//   re-expand if a shorter path appears) rather than mark-on-discovery.  In a
//   0/1-weighted graph a node first reached via a 1-edge from a dist-d node can
//   later be reached via a 0-edge from another dist-d node; mark-on-discovery
//   would lock it at d+1 and OVERESTIMATE -> inadmissible.  Relaxation is safe.
//
// MEMORY (build time, uint8 distance array over the blank-aware space):
//   Group A (k=7): 16*..*9  =   518,918,400  (~0.5 GB)
//   Group B (k=8): 16*..*8  = 4,151,347,200  (~4.1 GB)   + deque frontier
//
// USAGE:
//   build_pdbs --out <dir>     ->  writes pdb_a.bin, pdb_b.bin
// ---------------------------------------------------------------------------
#include "pdb.h"

#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace puzzle;

namespace {

std::vector<int> group_tiles(int group_id) {
    std::vector<int> g;
    for (int t = 1; t < N_CELLS; ++t)
        if (group_of(t) == group_id) g.push_back(t);
    return g;
}

// Lehmer rank of the first `n` cell positions in `pos` (mixed radix 16,15,...).
uint64_t rank_pos(const int* pos, int n) {
    uint32_t used = 0;
    uint64_t r = 0;
    for (int i = 0; i < n; ++i) {
        const int cell = pos[i];
        const uint32_t below = used & ((1u << cell) - 1u);
        r = r * static_cast<uint64_t>(N_CELLS - i)
            + static_cast<uint64_t>(cell - __builtin_popcount(below));
        used |= (1u << cell);
    }
    return r;
}

// Inverse of rank_pos for `n` positions: reconstruct cell indices into out_pos.
void decode_pos(uint64_t r, int n, int* out_pos) {
    int smalls[N_CELLS];
    for (int i = n - 1; i >= 0; --i) {
        const uint64_t base = static_cast<uint64_t>(N_CELLS - i);
        smalls[i] = static_cast<int>(r % base);
        r /= base;
    }
    uint32_t used = 0;
    for (int i = 0; i < n; ++i) {
        int count = 0;
        for (int c = 0; c < N_CELLS; ++c) {
            if (!(used & (1u << c))) {
                if (count++ == smalls[i]) {
                    out_pos[i] = c;
                    used |= (1u << c);
                    break;
                }
            }
        }
    }
}

void build_group(AdditivePDB& pdb) {
    const auto& group = pdb.group();
    const int   k     = static_cast<int>(group.size());
    const int   kp1   = k + 1;  // group tiles + blank

    uint64_t bfs_sz = 1;
    for (int i = 0; i < kp1; ++i) bfs_sz *= static_cast<uint64_t>(N_CELLS - i);

    std::vector<uint8_t> bfs_tbl(bfs_sz, 0xFFu);  // 0xFF = unvisited / infinity
    auto& pdb_tbl = pdb.table();                  // init_group'd to 0xFF, size pdb_size(k)

    // Goal blank-aware node: tile group[i] at cell group[i], blank at cell 0.
    int goal_pos[N_CELLS];
    for (int i = 0; i < k; ++i) goal_pos[i] = group[i];
    goal_pos[k] = 0;

    const uint64_t goal_r = rank_pos(goal_pos, kp1);
    bfs_tbl[goal_r]               = 0;
    pdb_tbl[rank_pos(goal_pos, k)] = 0;

    std::deque<uint64_t> dq;
    dq.push_back(goal_r);

    uint64_t processed = 0;
    while (!dq.empty()) {
        const uint64_t r = dq.front();
        dq.pop_front();
        const uint8_t c = bfs_tbl[r];

        int pos[N_CELLS];
        decode_pos(r, kp1, pos);
        const int blank = pos[k];

        uint16_t gcells = 0;  // bitmask of cells holding a group tile
        for (int i = 0; i < k; ++i) gcells |= static_cast<uint16_t>(1u << pos[i]);

        const int ncnt = NEIGHBOR_COUNT[blank];
        for (int ni = 0; ni < ncnt; ++ni) {
            const int  nb       = NEIGHBORS[blank][ni];
            const bool is_group = (gcells >> nb) & 1u;

            int np[N_CELLS];
            std::memcpy(np, pos, kp1 * sizeof(int));
            if (is_group) {
                int gi = 0;
                while (np[gi] != nb) ++gi;  // the group tile being slid
                np[gi] = blank;             // tile moves into old blank cell
            }
            np[k] = nb;                     // blank moves to neighbor cell

            const uint8_t  nc = static_cast<uint8_t>(c + (is_group ? 1 : 0));
            const uint64_t nr = rank_pos(np, kp1);
            if (nc < bfs_tbl[nr]) {
                bfs_tbl[nr] = nc;
                const uint64_t kr = rank_pos(np, k);
                if (nc < pdb_tbl[kr]) pdb_tbl[kr] = nc;
                if (is_group) dq.push_back(nr);
                else          dq.push_front(nr);
            }
        }
        if (++processed % 50'000'000ull == 0)
            std::cerr << "[pdb]   expanded " << processed
                      << "  deque=" << dq.size() << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string out_dir = ".";
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_dir = argv[++i];

    std::filesystem::create_directories(out_dir);

    {
        AdditivePDB a;
        a.init_group(group_tiles(0));
        std::cerr << "[pdb] building group A (" << a.group().size() << " tiles)...\n";
        build_group(a);
        const std::string p = out_dir + "/pdb_a.bin";
        if (!a.save(p)) { std::cerr << "[pdb] save failed: " << p << "\n"; return 1; }
        std::cerr << "[pdb] group A saved -> " << p << "\n";
    }
    {
        AdditivePDB b;
        b.init_group(group_tiles(1));
        std::cerr << "[pdb] building group B (" << b.group().size() << " tiles)...\n";
        build_group(b);
        const std::string p = out_dir + "/pdb_b.bin";
        if (!b.save(p)) { std::cerr << "[pdb] save failed: " << p << "\n"; return 1; }
        std::cerr << "[pdb] group B saved -> " << p << "\n";
    }
    return 0;
}
