#include "pdb.h"
#include <fstream>
#include <stdexcept>

namespace puzzle {

// File format (little-endian):
//   uint32 magic = 0x50444231 ("PDB1")
//   uint32 group_size  k
//   int32  group_tiles[k]
//   uint64 table_size  = pdb_size(k)
//   uint8  table[table_size]
static constexpr uint32_t PDB_MAGIC = 0x50444231u;

void AdditivePDB::init_group(std::vector<int> group_tiles) {
    group_ = std::move(group_tiles);
    table_.assign(pdb_size(static_cast<int>(group_.size())), 0xFF); // 0xFF = unfilled
}

// Partial-permutation (Lehmer) rank of the group's tile positions.
//   For each group tile (in fixed order) take its cell, count how many
//   not-yet-used cells are smaller, and fold into a mixed-radix number with
//   bases 16, 15, 14, ...  -> dense index in [0, 16*15*...*(16-k+1)).
uint64_t AdditivePDB::rank(State s) const {
    // cell position of each tile value, one scan.
    int pos[N_CELLS];
    for (int p = 0; p < N_CELLS; ++p) pos[tile_at(s, p)] = p;

    uint32_t used = 0; // bitmask of consumed cells
    uint64_t r = 0;
    int i = 0;
    for (int tile : group_) {
        const int cell = pos[tile];
        // count unused cells with index < cell
        const uint32_t below = used & ((1u << cell) - 1u);
        const int smaller = cell - __builtin_popcount(below);
        r = r * static_cast<uint64_t>(N_CELLS - i) + static_cast<uint64_t>(smaller);
        used |= (1u << cell);
        ++i;
    }
    return r;
}

bool AdditivePDB::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    uint32_t magic = 0, k = 0;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&k), 4);
    if (magic != PDB_MAGIC || k == 0 || k > N_CELLS) return false;

    group_.resize(k);
    for (uint32_t i = 0; i < k; ++i) {
        int32_t t = 0;
        f.read(reinterpret_cast<char*>(&t), 4);
        group_[i] = t;
    }

    uint64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 8);
    if (n != pdb_size(static_cast<int>(k))) return false;

    table_.resize(n);
    f.read(reinterpret_cast<char*>(table_.data()), static_cast<std::streamsize>(n));
    return static_cast<bool>(f);
}

bool AdditivePDB::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    const uint32_t magic = PDB_MAGIC;
    const uint32_t k = static_cast<uint32_t>(group_.size());
    f.write(reinterpret_cast<const char*>(&magic), 4);
    f.write(reinterpret_cast<const char*>(&k), 4);
    for (int t : group_) {
        const int32_t v = t;
        f.write(reinterpret_cast<const char*>(&v), 4);
    }
    const uint64_t n = table_.size();
    f.write(reinterpret_cast<const char*>(&n), 8);
    f.write(reinterpret_cast<const char*>(table_.data()), static_cast<std::streamsize>(n));
    return static_cast<bool>(f);
}

} // namespace puzzle
