#pragma once
#include "puzzle_defs.h"
#include "puzzle_state.h"
#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Heuristics for IDA*.  Every heuristic exposes `int h(State, int blank)`.
//
//  * ManhattanHeuristic — admissible placeholder, works with no setup.  Used
//    until the additive PDBs are built.  Too weak for deep (1000-move) random
//    states at scale, but correct.
//
//  * AdditivePDB / SumHeuristic — disjoint pattern databases whose costs are
//    SUMMED (admissible because the groups' move sets are disjoint).  This is
//    the production heuristic; build the tables once with `build_pdbs`.
// ---------------------------------------------------------------------------
namespace puzzle {

struct ManhattanHeuristic {
    int h(State s, int /*blank*/) const { return manhattan(s); }
};

// One additive pattern database for a disjoint group of tiles.
class AdditivePDB {
public:
    // --- query path (used by generate_data) ---
    bool load(const std::string& path);
    int  lookup(State s) const { return table_[rank(s)]; }
    bool ready() const { return !table_.empty(); }

    // --- build path (used by build_pdbs) ---
    void init_group(std::vector<int> group_tiles); // sets group_, sizes table_
    bool save(const std::string& path) const;
    std::vector<uint8_t>&       table()       { return table_; }
    const std::vector<uint8_t>& table() const { return table_; }
    const std::vector<int>&     group() const { return group_; }

    // Partial-permutation rank of this group's tile positions in `s`.
    // Result is a dense index in [0, pdb_size(group_.size())).
    uint64_t rank(State s) const;

private:
    std::vector<int>     group_;  // tile ids, fixed order (defines the index)
    std::vector<uint8_t> table_;  // table_[rank] = # moves of group tiles to solve
};

// Additive heuristic: SUM of two disjoint PDB lookups.
struct SumHeuristic {
    AdditivePDB a, b;
    int h(State s, int /*blank*/) const { return a.lookup(s) + b.lookup(s); }
    bool ready() const { return a.ready() && b.ready(); }
};

} // namespace puzzle
