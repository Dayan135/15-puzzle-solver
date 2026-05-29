#include "move_tables.h"
#include <cstring>

// ---------------------------------------------------------------------------
// Quarter-turn base data (Kociemba convention)
//
// Corner indices: 0=URF 1=UFL 2=ULB 3=UBR 4=DFR 5=DLF 6=DBL 7=DRB
// Edge indices:   0=UR  1=UF  2=UL  3=UB  4=DR  5=DF  6=DL  7=DB
//                 8=FR  9=FL 10=BL 11=BR
//
// For each of the 6 face quarter-turns the tables express:
//   cp[old_pos] = new_pos    (where does a cubie at old_pos go?)
//   co[old_pos] = delta_ori  (how much does its orientation change, mod 3?)
//   ep[old_pos] = new_pos
//   ef[old_pos] = flip_xor
//
// Index mapping to Move enum: U=0,D=1,L=2,R=3,F=4,B=5  (face index)
// We'll store them indexed 0..5 here, then map to the 18-move enum.
// ---------------------------------------------------------------------------
namespace {

// ---- U face (CW from top): cycle 0→1→2→3→0, no twist, no flip ----------
constexpr uint8_t cpU[8]  = {1,2,3,0, 4,5,6,7};
constexpr uint8_t coU[8]  = {0,0,0,0, 0,0,0,0};
constexpr uint8_t epU[12] = {1,2,3,0, 4,5,6,7, 8,9,10,11};
constexpr uint8_t efU[12] = {0,0,0,0, 0,0,0,0, 0,0, 0, 0};

// ---- D face (CW from bottom): cycle 4→5→6→7→4, no twist, no flip -------
constexpr uint8_t cpD[8]  = {0,1,2,3, 5,6,7,4};
constexpr uint8_t coD[8]  = {0,0,0,0, 0,0,0,0};
constexpr uint8_t epD[12] = {0,1,2,3, 5,6,7,4, 8,9,10,11};
constexpr uint8_t efD[12] = {0,0,0,0, 0,0,0,0, 0,0, 0, 0};

// ---- L face (CW from left): cycle 1→2→6→5→1, twist +1,+2,+1,+2 ---------
// U→B, B→D, D→F, F→U  (left layer)
// Edge cycle: UL(2)→BL(10)→DL(6)→FL(9)→UL(2), no flip
constexpr uint8_t cpL[8]  = {0,2,6,3, 4,1,5,7};
constexpr uint8_t coL[8]  = {0,1,2,0, 0,2,1,0};
constexpr uint8_t epL[12] = {0,1,10,3, 4,5,9,7, 8,2,6,11};
constexpr uint8_t efL[12] = {0,0, 0,0, 0,0,0,0, 0,0,0, 0};

// ---- R face (CW from right): cycle 0→4→7→3→0, twist +2,+1,+2,+1 --------
// U→F, F→D, D→B, B→U  (right layer)
// Edge cycle: UR(0)→FR(8)→DR(4)→BR(11)→UR(0), no flip
constexpr uint8_t cpR[8]  = {4,1,2,0, 7,5,6,3};
constexpr uint8_t coR[8]  = {2,0,0,1, 1,0,0,2};
constexpr uint8_t epR[12] = {8,1,2,3, 11,5,6,7, 4,9,10,0};
constexpr uint8_t efR[12] = {0,0,0,0,  0,0,0,0, 0,0, 0,0};

// ---- F face (CW from front): cycle 0→4→5→1→0, twist +1,+2,+2,+1 --------
// U→R, R→D, D→L, L→U  (front layer)
// Edge cycle: UF(1)→FR(8)→DF(5)→FL(9)→UF(1), all flipped
constexpr uint8_t cpF[8]  = {4,0,2,3, 5,1,6,7};
constexpr uint8_t coF[8]  = {1,2,0,0, 2,1,0,0};
constexpr uint8_t epF[12] = {0,8,2,3, 4,9,6,7, 5,1,10,11};
constexpr uint8_t efF[12] = {0,1,0,0, 0,1,0,0, 1,1, 0, 0};

// ---- B face (CW from back): cycle 3→2→6→7→3, twist +2,+1,+2,+1 ---------
// U→L, L→D, D→R, R→U  (back layer)
// Edge cycle: UB(3)→BL(10)→DB(7)→BR(11)→UB(3), all flipped
constexpr uint8_t cpB[8]  = {0,1,6,2, 4,5,7,3};
constexpr uint8_t coB[8]  = {0,0,1,2, 0,0,2,1};
constexpr uint8_t epB[12] = {0,1,2,10, 4,5,6,11, 8,9,7,3};
constexpr uint8_t efB[12] = {0,0,0, 1, 0,0,0, 1, 0,0,1,1};

// Pointers to the 6 quarter-turn tables (face index 0..5)
const uint8_t* const QCP[6] = {cpU,cpD,cpL,cpR,cpF,cpB};
const uint8_t* const QCO[6] = {coU,coD,coL,coR,coF,coB};
const uint8_t* const QEP[6] = {epU,epD,epL,epR,epF,epB};
const uint8_t* const QEF[6] = {efU,efD,efL,efR,efF,efB};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Composition helpers
// ---------------------------------------------------------------------------

// new_corner_pos[c] = pb[ pa[c] ]   (apply a then b)
void MoveTables::compose_corner_pos(const uint8_t a[8], const uint8_t b[8], uint8_t out[8]) {
    for (int i = 0; i < 8; ++i) out[i] = b[a[i]];
}

// Apply move A then move B to corners (both pos+ori together)
void MoveTables::compose_corner_ori(
    const uint8_t pa[8], const uint8_t oa[8],
    const uint8_t pb[8], const uint8_t ob[8],
    uint8_t out_pos[8],  uint8_t out_ori[8])
{
    for (int i = 0; i < 8; ++i) {
        out_pos[i] = pb[pa[i]];
        out_ori[i] = static_cast<uint8_t>((oa[i] + ob[pa[i]]) % 3u);
    }
}

void MoveTables::compose_edge_pos(const uint8_t a[12], const uint8_t b[12], uint8_t out[12]) {
    for (int i = 0; i < 12; ++i) out[i] = b[a[i]];
}

// Apply move A then move B to edges (both pos+flip together)
void MoveTables::compose_edge_full(
    const uint8_t pa[12], const uint8_t fa[12],
    const uint8_t pb[12], const uint8_t fb[12],
    uint8_t out_pos[12],  uint8_t out_flip[12])
{
    for (int i = 0; i < 12; ++i) {
        out_pos[i]  = pb[pa[i]];
        out_flip[i] = static_cast<uint8_t>(fa[i] ^ fb[pa[i]]);
    }
}

// ---------------------------------------------------------------------------
// MoveTables constructor — builds all 18 move tables
//
// Move enum layout: face*3 + {0=quarter CW, 1=half, 2=quarter CCW}
// So face index = move / 3, type = move % 3
// ---------------------------------------------------------------------------
MoveTables::MoveTables() {
    for (int face = 0; face < 6; ++face) {
        const uint8_t* cp1 = QCP[face];
        const uint8_t* co1 = QCO[face];
        const uint8_t* ep1 = QEP[face];
        const uint8_t* ef1 = QEF[face];

        int m_cw   = face * 3;      // quarter CW  (e.g. U=0)
        int m_half = face * 3 + 1;  // half turn    (e.g. U2=1)
        int m_ccw  = face * 3 + 2;  // quarter CCW  (e.g. UP=2)

        // Quarter CW: copy directly
        std::memcpy(corner_pos[m_cw], cp1, 8);
        std::memcpy(corner_ori[m_cw], co1, 8);
        std::memcpy(edge_pos  [m_cw], ep1, 12);
        std::memcpy(edge_flip [m_cw], ef1, 12);

        // Half turn: CW composed with CW
        uint8_t cp2[8], co2[8], ep2[12], ef2[12];
        compose_corner_ori(cp1, co1, cp1, co1, cp2, co2);
        compose_edge_full (ep1, ef1, ep1, ef1, ep2, ef2);
        std::memcpy(corner_pos[m_half], cp2, 8);
        std::memcpy(corner_ori[m_half], co2, 8);
        std::memcpy(edge_pos  [m_half], ep2, 12);
        std::memcpy(edge_flip [m_half], ef2, 12);

        // Quarter CCW: CW composed with itself three times = CW then half
        uint8_t cp3[8], co3[8], ep3[12], ef3[12];
        compose_corner_ori(cp1, co1, cp2, co2, cp3, co3);
        compose_edge_full (ep1, ef1, ep2, ef2, ep3, ef3);
        std::memcpy(corner_pos[m_ccw], cp3, 8);
        std::memcpy(corner_ori[m_ccw], co3, 8);
        std::memcpy(edge_pos  [m_ccw], ep3, 12);
        std::memcpy(edge_flip [m_ccw], ef3, 12);
    }
}

const MoveTables& MoveTables::get() {
    static const MoveTables instance;
    return instance;
}
