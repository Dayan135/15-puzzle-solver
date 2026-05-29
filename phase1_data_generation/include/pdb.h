#pragma once
#include "cube_state.h"
#include <vector>
#include <string>
#include <cstdint>

// ---------------------------------------------------------------------------
// PatternDB — nibble-packed pattern database
//
// Stores one 4-bit value per state index.
// data_[i/2]: low nibble = entry i, high nibble = entry i+1
// Entries initialized to 0xFF (unvisited = 15) by the builder.
// ---------------------------------------------------------------------------
class PatternDB {
public:
    explicit PatternDB(std::size_t num_entries);

    // Load from / save to a binary file.
    // Format: 8-byte header (magic + entry count) followed by raw nibble data.
    void load(const std::string& path);
    void save(const std::string& path) const;

    // Returns the 4-bit heuristic value stored at index.
    inline uint8_t get(uint32_t index) const noexcept {
        uint8_t byte = data_[index >> 1];
        return (index & 1u) ? (byte >> 4) : (byte & 0x0Fu);
    }

    // Store a 4-bit value (val must be <= 15).
    inline void set(uint32_t index, uint8_t val) noexcept {
        uint8_t& byte = data_[index >> 1];
        if (index & 1u)
            byte = static_cast<uint8_t>((byte & 0x0Fu) | (val << 4));
        else
            byte = static_cast<uint8_t>((byte & 0xF0u) | (val & 0x0Fu));
    }

    std::size_t num_entries() const noexcept { return size_; }

private:
    std::vector<uint8_t> data_; // ceil(size_ / 2) bytes
    std::size_t          size_;
};

// ---------------------------------------------------------------------------
// MaxHeuristic — admissible heuristic: max(corner, edge_A, edge_B)
// ---------------------------------------------------------------------------
class MaxHeuristic {
public:
    MaxHeuristic(const PatternDB& corner,
                 const PatternDB& edge_a,
                 const PatternDB& edge_b);

    uint8_t operator()(const CubeState& s) const noexcept;

private:
    const PatternDB& corner_;
    const PatternDB& edge_a_;
    const PatternDB& edge_b_;
};
