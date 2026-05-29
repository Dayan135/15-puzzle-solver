#pragma once
#include <cstdint>

// ---------------------------------------------------------------------------
// Move enumeration — 18 standard Rubik's Cube moves
// Face order:  U=0, D=1, L=2, R=3, F=4, B=5
// Move order within each face: quarter CW, half turn, quarter CCW
// ---------------------------------------------------------------------------
enum Move : int {
    U = 0, U2 = 1, UP = 2,   // Up face
    D = 3, D2 = 4, DP = 5,   // Down face
    L = 6, L2 = 7, LP = 8,   // Left face
    R = 9, R2 = 10, RP = 11, // Right face
    F = 12, F2 = 13, FP = 14,// Front face
    B = 15, B2 = 16, BP = 17 // Back face
};

constexpr int N_MOVES   = 18;
constexpr int N_CORNERS = 8;
constexpr int N_EDGES   = 12;
constexpr int N_FACES   = 6;

// Sentinel values for IDA* search
constexpr int IDA_FOUND = -1;
constexpr int IDA_INF   = 100;
constexpr int NO_MOVE   = -1;

// Inverse of each move: inverse[m] gives the move that undoes m
constexpr int MOVE_INVERSE[N_MOVES] = {
    2, 1, 0,    // U  U2  U'
    5, 4, 3,    // D  D2  D'
    8, 7, 6,    // L  L2  L'
   11,10, 9,    // R  R2  R'
   14,13,12,    // F  F2  F'
   17,16,15     // B  B2  B'
};

// Which face (0-5) each move belongs to
constexpr int MOVE_FACE[N_MOVES] = {
    0,0,0,   // U, U2, U'
    1,1,1,   // D, D2, D'
    2,2,2,   // L, L2, L'
    3,3,3,   // R, R2, R'
    4,4,4,   // F, F2, F'
    5,5,5    // B, B2, B'
};

// Opposite face for each face: U<->D, L<->R, F<->B
constexpr int OPPOSITE_FACE[N_FACES] = {1, 0, 3, 2, 5, 4};

// ---------------------------------------------------------------------------
// Corner and edge index conventions (Kociemba standard)
//
// Corners: 0=URF  1=UFL  2=ULB  3=UBR  4=DFR  5=DLF  6=DBL  7=DRB
// Edges:   0=UR   1=UF   2=UL   3=UB   4=DR   5=DF   6=DL   7=DB
//          8=FR   9=FL  10=BL  11=BR
// ---------------------------------------------------------------------------

// 6-edge PDB partitions (disjoint, covering all 12 edges)
// Set A: upper-layer + front edges
constexpr int EDGE_SET_A[6] = {0, 1, 2, 3, 8, 9};   // UR,UF,UL,UB,FR,FL
// Set B: lower-layer + back edges
constexpr int EDGE_SET_B[6] = {4, 5, 6, 7, 10, 11}; // DR,DF,DL,DB,BL,BR

// PDB sizes
constexpr uint32_t CORNER_PDB_SIZE = 88179840u; // 8! * 3^7
constexpr uint32_t EDGE_PDB_SIZE   = 42577920u; // P(12,6) * 2^6
