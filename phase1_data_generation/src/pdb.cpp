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
    // Corner PDB index
    uint32_t ci = corner_perm_rank(s) * 2187u + corner_ori_rank(s);
    //arrangement of 8 corners → [0, 40319] * 2187 + orientation of 8 corners → [0, 2186]

    // 6-edge PDB A index
    uint32_t ea_idx = k_perm_rank(s, EDGE_SET_A, 6) * 64u + k_flip_rank(s, EDGE_SET_A, 6);

    // 6-edge PDB B index
    uint32_t eb_idx = k_perm_rank(s, EDGE_SET_B, 6) * 64u + k_flip_rank(s, EDGE_SET_B, 6);

    uint8_t hc = corner_.get(ci);
    uint8_t ha = edge_a_.get(ea_idx);
    uint8_t hb = edge_b_.get(eb_idx);

    return std::max({hc, ha, hb});
}
