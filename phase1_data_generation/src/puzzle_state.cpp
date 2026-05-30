#include "puzzle_state.h"

namespace puzzle {

// ---------------------------------------------------------------------------
// Neighbor tables, built once at static-init time from the 4x4 geometry.
// ---------------------------------------------------------------------------
static std::array<std::array<int8_t, 4>, N_CELLS> make_neighbors() {
    std::array<std::array<int8_t, 4>, N_CELLS> nb{};
    for (auto& row : nb) row.fill(-1);
    for (int c = 0; c < N_CELLS; ++c) {
        const int r = row_of(c), col = col_of(c);
        int k = 0;
        if (r > 0)          nb[c][k++] = static_cast<int8_t>(c - SIDE); // up
        if (r < SIDE - 1)   nb[c][k++] = static_cast<int8_t>(c + SIDE); // down
        if (col > 0)        nb[c][k++] = static_cast<int8_t>(c - 1);    // left
        if (col < SIDE - 1) nb[c][k++] = static_cast<int8_t>(c + 1);    // right
    }
    return nb;
}

static std::array<int8_t, N_CELLS> make_counts(
        const std::array<std::array<int8_t, 4>, N_CELLS>& nb) {
    std::array<int8_t, N_CELLS> cnt{};
    for (int c = 0; c < N_CELLS; ++c) {
        int k = 0;
        while (k < 4 && nb[c][k] >= 0) ++k;
        cnt[c] = static_cast<int8_t>(k);
    }
    return cnt;
}

const std::array<std::array<int8_t, 4>, N_CELLS> NEIGHBORS      = make_neighbors();
const std::array<int8_t, N_CELLS>                NEIGHBOR_COUNT = make_counts(NEIGHBORS);

// ---------------------------------------------------------------------------
State scramble(State start, int n_moves, std::mt19937_64& rng) {
    State s = start;
    int blank = find_blank(s);
    int prev  = -1; // cell the blank came from; never slide straight back
    for (int i = 0; i < n_moves; ++i) {
        const int cnt = NEIGHBOR_COUNT[blank]; // always >= 2, so >=1 non-reverse option
        int n;
        do {
            n = NEIGHBORS[blank][rng() % cnt];
        } while (n == prev);
        s     = slide(s, blank, n);
        prev  = blank;
        blank = n;
    }
    return s;
}

// ---------------------------------------------------------------------------
bool is_solvable(State s) {
    int a[N_CELLS];
    int bpos = 0;
    for (int p = 0; p < N_CELLS; ++p) {
        a[p] = tile_at(s, p);
        if (a[p] == 0) bpos = p;
    }
    // Parity of the full 16-symbol permutation (blank = symbol 0).
    int inv = 0;
    for (int i = 0; i < N_CELLS; ++i)
        for (int j = i + 1; j < N_CELLS; ++j)
            if (a[i] > a[j]) ++inv;
    // Each slide flips both the permutation parity and the blank's taxicab
    // distance to its home cell (cell 0), so the XOR is invariant.
    const int blank_taxi = row_of(bpos) + col_of(bpos);
    return (inv & 1) == (blank_taxi & 1);
}

// ---------------------------------------------------------------------------
int manhattan(State s) {
    int d = 0;
    for (int p = 0; p < N_CELLS; ++p) {
        const int t = tile_at(s, p);
        if (t == 0) continue;
        int dr = row_of(p) - row_of(t);
        int dc = col_of(p) - col_of(t);
        d += (dr < 0 ? -dr : dr) + (dc < 0 ? -dc : dc);
    }
    return d;
}

} // namespace puzzle
