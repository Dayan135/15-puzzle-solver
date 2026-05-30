#pragma once
#include "move_tables.h" // CubeState, N_CORNERS, N_EDGES

// ---------------------------------------------------------------------------
// SymmetryTables — the 48 rigid symmetries of the cube (24 rotations + 24
// mirror reflections).
//
// The symmetry-enhanced heuristic queries h over all 48 conjugates of a state:
//     h_sym(s) = max over σ of  h( σ · s · σ⁻¹ )
// Because conjugation by a cube symmetry preserves optimal solution length,
// every term is an admissible lower bound, and the max is far tighter than a
// single PDB lookup (pushes h from ~9 to ~16 for random states).
//
// A CubeState is itself a cube-group element in the SAME representation used by
// moves: arrays are indexed by a cubie's home position, the value is its
// current (position<<k)|orientation. So we conjugate by composing group
// elements directly.
//
// Each symmetry is stored as a full group element `Elem` together with its
// inverse, so conjugation is two compositions.
// ---------------------------------------------------------------------------
struct SymElem {
    uint8_t cp[N_CORNERS];  // corner position permutation (home pos -> current pos)
    uint8_t co[N_CORNERS];  // corner orientation contribution (mod 3)
    uint8_t cm;             // corner orientation multiplier: 1 (rotation) or 2 (reflection)
    uint8_t ep[N_EDGES];    // edge position permutation
    uint8_t ef[N_EDGES];    // edge flip contribution (XOR)
};

struct SymmetryTables {
    // 16 U/D-axis-preserving symmetries (rotations about U-D, U/D swap, L-R
    // mirror). These keep the corner-orientation reference axis fixed, so the
    // orientation transform is a simple uniform multiplier — provably correct.
    // (The full 48-group needs position-dependent twist tables for the body-
    //  diagonal rotations; deferred unless 16 proves insufficient.)
    static constexpr int N_SYM = 16;

    SymElem sym[N_SYM];     // σ
    SymElem inv[N_SYM];     // σ⁻¹

    static const SymmetryTables& get();

private:
    SymmetryTables();
};

// Compose two cube-group elements: result = "do A, then B"  (i.e. B∘A).
// Corner orientation under reflection uses the multiplier cm.
inline SymElem sym_compose(const SymElem& A, const SymElem& B) noexcept {
    SymElem r;
    for (int c = 0; c < N_CORNERS; ++c) {
        r.cp[c] = B.cp[A.cp[c]];
        r.co[c] = static_cast<uint8_t>((B.cm * A.co[c] + B.co[A.cp[c]]) % 3u);
    }
    r.cm = static_cast<uint8_t>((A.cm * B.cm) % 3u);
    for (int e = 0; e < N_EDGES; ++e) {
        r.ep[e] = B.ep[A.ep[e]];
        r.ef[e] = static_cast<uint8_t>(A.ef[e] ^ B.ef[A.ep[e]]);
    }
    return r;
}

// ---------------------------------------------------------------------------
// apply_symmetry — returns the conjugate  σ · s · σ⁻¹  as a CubeState.
//
// A CubeState maps directly onto a SymElem with cm == 1 (no reflection),
// co = corner orientation, ef = edge flip.
// Conjugation in "do-then" composition order is:
//     σ · s · σ⁻¹  =  compose( compose(σ⁻¹, s), σ )
// ---------------------------------------------------------------------------
inline CubeState apply_symmetry(const CubeState& s, int i,
                                const SymmetryTables& st) noexcept {
    SymElem g;
    for (int c = 0; c < N_CORNERS; ++c) {
        g.cp[c] = static_cast<uint8_t>(s.corners[c] >> 2);
        g.co[c] = static_cast<uint8_t>(s.corners[c] & 3u);
    }
    g.cm = 1;
    for (int e = 0; e < N_EDGES; ++e) {
        g.ep[e] = static_cast<uint8_t>(s.edges[e] >> 1);
        g.ef[e] = static_cast<uint8_t>(s.edges[e] & 1u);
    }

    const SymElem t = sym_compose(sym_compose(st.inv[i], g), st.sym[i]);

    CubeState r;
    for (int c = 0; c < N_CORNERS; ++c)
        r.corners[c] = static_cast<uint8_t>((t.cp[c] << 2) | t.co[c]);
    for (int e = 0; e < N_EDGES; ++e)
        r.edges[e] = static_cast<uint8_t>((t.ep[e] << 1) | t.ef[e]);
    return r;
}
