#include "pdb.h"
#include "cube_defs.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cstring>

// ---------------------------------------------------------------------------
// PatternDB
// ---------------------------------------------------------------------------

PatternDB::PatternDB(std::size_t num_entries)
    : size_(num_entries)
{
    std::size_t bytes = (num_entries + 1) / 2;
    data_.assign(bytes, 0xFFu); // 0xFF → both nibbles = 15 = unvisited
}

static constexpr uint32_t PDB_MAGIC   = 0x50444221u; // "PDB!"
static constexpr uint32_t PDB_VERSION = 1u;

void PatternDB::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open PDB file: " + path);

    uint32_t magic, version;
    uint64_t n_entries;
    f.read(reinterpret_cast<char*>(&magic),    4);
    f.read(reinterpret_cast<char*>(&version),  4);
    f.read(reinterpret_cast<char*>(&n_entries),8);

    if (magic != PDB_MAGIC)
        throw std::runtime_error("Bad PDB magic in: " + path);
    if (n_entries != size_)
        throw std::runtime_error("PDB entry count mismatch in: " + path);

    std::size_t bytes = (size_ + 1) / 2;
    f.read(reinterpret_cast<char*>(data_.data()), static_cast<std::streamsize>(bytes));
    if (!f) throw std::runtime_error("Truncated PDB file: " + path);
}

void PatternDB::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write PDB file: " + path);

    uint32_t magic   = PDB_MAGIC;
    uint32_t version = PDB_VERSION;
    uint64_t n_ent   = static_cast<uint64_t>(size_);
    f.write(reinterpret_cast<const char*>(&magic),   4);
    f.write(reinterpret_cast<const char*>(&version), 4);
    f.write(reinterpret_cast<const char*>(&n_ent),   8);

    std::size_t bytes = (size_ + 1) / 2;
    f.write(reinterpret_cast<const char*>(data_.data()), static_cast<std::streamsize>(bytes));
}

// ---------------------------------------------------------------------------
// MaxHeuristic
// ---------------------------------------------------------------------------

MaxHeuristic::MaxHeuristic(const PatternDB& corner,
                           const PatternDB& edge_a,
                           const PatternDB& edge_b)
    : corner_(corner), edge_a_(edge_a), edge_b_(edge_b) {}

uint8_t MaxHeuristic::operator()(const CubeState& s) const noexcept {
    // Plain 3-PDB max heuristic: max(corner, edge_A, edge_B). Admissible.
    //
    // NOTE: a symmetry-enhanced variant (max over the 16 U/D-preserving
    // conjugates, see symmetry.h) was implemented and verified correct, but
    // benchmarks showed it ~4x SLOWER here: these PDBs cap at depth ~11, so the
    // max-over-symmetries cannot exceed that ceiling and yields almost no extra
    // pruning while costing 16x the lookups. Kept in the tree for reference.
    const uint32_t ci = corner_perm_rank(s) * 2187u + corner_ori_rank(s);
    const uint32_t ea = k_perm_rank(s, EDGE_SET_A, 6) * 64u + k_flip_rank(s, EDGE_SET_A, 6);
    const uint32_t eb = k_perm_rank(s, EDGE_SET_B, 6) * 64u + k_flip_rank(s, EDGE_SET_B, 6);

    return std::max({corner_.get(ci), edge_a_.get(ea), edge_b_.get(eb)});
}
