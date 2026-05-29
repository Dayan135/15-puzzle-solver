#include "cube_state.h"
#include <cstring>
#include <cassert>

CubeState solved_state() noexcept {
    CubeState s;
    for (int c = 0; c < N_CORNERS; ++c) s.corners[c] = static_cast<uint8_t>(c << 2);
    for (int e = 0; e < N_EDGES;   ++e) s.edges[e]   = static_cast<uint8_t>(e << 1);
    return s;
}

bool is_solved(const CubeState& s) noexcept {
    const CubeState sol = solved_state();
    return std::memcmp(s.corners, sol.corners, N_CORNERS) == 0 &&
           std::memcmp(s.edges,   sol.edges,   N_EDGES)   == 0;
}

CubeState random_state(std::mt19937& rng) noexcept {
    const MoveTables& mt = MoveTables::get();
    CubeState s = solved_state();
    int last_face = -1;

    // 100-move walk; skip the inverse of the previous move to avoid
    // trivially undoing one step, which improves mixing without biasing.
    for (int step = 0; step < 100; ++step) {
        // Build candidate move list: exclude moves on the same face as last move
        // and the opposite face (to avoid U then D, which can be reordered).
        // A simple and correct approach: just exclude the direct inverse.
        int m;
        do {
            m = static_cast<int>(rng() % N_MOVES);
        } while (last_face >= 0 && MOVE_FACE[m] == last_face);
        s = apply_move(s, m, mt);
        last_face = MOVE_FACE[m];
    }
    return s;
}

// ---------------------------------------------------------------------------
// Lehmer-code rank of the corner permutation.
// corners[c] >> 2 gives the slot (position) of cubie c.
// We rank the sequence [pos(0), pos(1), ..., pos(7)] as a permutation of {0..7}.
// ---------------------------------------------------------------------------
uint32_t corner_perm_rank(const CubeState& s) noexcept {
    // Extract positions
    uint8_t pos[N_CORNERS];
    for (int c = 0; c < N_CORNERS; ++c) pos[c] = s.corners[c] >> 2;

    uint32_t rank = 0;
    for (int i = 0; i < N_CORNERS - 1; ++i) {
        rank += pos[i];
        // Subtract count of pos[j<i] that are less than pos[i]
        for (int j = 0; j < i; ++j)
            if (pos[j] < pos[i]) --rank;
        // Scale by remaining factorial
        uint32_t fact = 1;
        for (int f = 1; f < N_CORNERS - 1 - i; ++f) fact *= static_cast<uint32_t>(f + 1);
        rank *= fact; // will be corrected: this is standard Lehmer factoriadic
    }

    // The above has a bug in the naive expansion. Use the correct factoriadic:
    // rank = sum_i  ( lehmer_i * (n-1-i)! )
    // Recompute properly:
    rank = 0;
    static constexpr uint32_t FACT[8] = {1,1,2,6,24,120,720,5040};
    bool used[8] = {};
    for (int i = 0; i < N_CORNERS - 1; ++i) {
        uint8_t p = pos[i];
        uint32_t cnt = 0;
        for (int j = 0; j < static_cast<int>(p); ++j)
            if (!used[j]) ++cnt;
        rank += cnt * FACT[N_CORNERS - 1 - i];
        used[p] = true;
    }
    return rank;
}

// ---------------------------------------------------------------------------
// Base-3 rank of the first 7 corner orientations.
// The 8th orientation is fully determined by the constraint sum mod 3 = 0.
// ---------------------------------------------------------------------------
uint32_t corner_ori_rank(const CubeState& s) noexcept {
    uint32_t rank = 0;
    for (int c = 0; c < N_CORNERS - 1; ++c) {
        rank = rank * 3u + (s.corners[c] & 3u);
    }
    return rank;
}

// ---------------------------------------------------------------------------
// k-permutation rank
//
// Given k edge cubies (identified by their cubie indices in edge_set[]),
// each currently sitting at some position among the 12 edge slots, compute
// the rank of that ordered partial permutation.
//
// The rank is computed as the "factoriadic" index of the ordered tuple
// (pos[0], pos[1], ..., pos[k-1]) where each pos[i] is in [0,11] and all
// positions are distinct.
//
// Equivalent to:
//   rank = sum_i  (number of positions[j>i] that are less than positions[i]
//                  AND not occupied by earlier elements) * P(n-i-1, k-i-1)
// ---------------------------------------------------------------------------
uint32_t k_perm_rank(const CubeState& s, const int* edge_set, int k) noexcept {
    // Collect the positions of the tracked edges
    uint8_t pos[12]; // at most 12 edges
    for (int i = 0; i < k; ++i)
        pos[i] = s.edges[edge_set[i]] >> 1;

    // Precompute P(n, r) = n! / (n-r)!
    // P(12,0)=1, P(12,1)=12, P(12,2)=132, ...
    // We need P(12-i-1, k-i-1) for i in [0, k-1]
    // Store as partial_perm[i] = P(n-i-1, k-i-1) where n=12
    static constexpr int N_SLOTS = 12;
    // Precompute: partial_perm[i] = product_{j=0}^{k-i-2} (n-i-1-j)
    //           = (n-i-1)*(n-i-2)*...*(n-k+1)   for i < k-1; = 1 for i=k-1
    uint32_t partial_perm[12];
    for (int i = 0; i < k; ++i) {
        uint32_t val = 1;
        for (int j = 0; j < k - i - 1; ++j)
            val *= static_cast<uint32_t>(N_SLOTS - i - 1 - j);
        partial_perm[i] = val;
    }

    uint32_t rank = 0;
    bool used[12] = {};
    for (int i = 0; i < k; ++i) {
        uint8_t p = pos[i];
        // Count how many unused positions < p come before p
        uint32_t cnt = 0;
        for (int j = 0; j < static_cast<int>(p); ++j)
            if (!used[j]) ++cnt;
        rank += cnt * partial_perm[i];
        used[p] = true;
    }
    return rank;
}

// ---------------------------------------------------------------------------
// Bit-packed flip rank: bit i = flip of edge_set[i]
// ---------------------------------------------------------------------------
uint8_t k_flip_rank(const CubeState& s, const int* edge_set, int k) noexcept {
    uint8_t rank = 0;
    for (int i = 0; i < k; ++i)
        rank = static_cast<uint8_t>(rank | ((s.edges[edge_set[i]] & 1u) << i));
    return rank;
}
