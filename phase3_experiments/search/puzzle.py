"""Vectorized 15-puzzle state operations on nibble-packed uint64 boards.

Mirrors phase1's encoding exactly (include/puzzle_state.h): nibble p holds the
tile at cell p, tile 0 is the blank, goal = 0xFEDCBA9876543210. All batch
functions take uint64[N] arrays and never loop over states in Python.
"""

from __future__ import annotations

import numpy as np

N_CELLS = 16
SIDE = 4
GOAL = 0xFEDCBA9876543210
MAX_COST = 80

_SHIFTS = np.arange(N_CELLS, dtype=np.uint64) * np.uint64(4)   # [16]
_CELL_OFFSETS = np.arange(N_CELLS, dtype=np.int64) * 16        # one-hot base per cell
_CELL_ROW = np.arange(N_CELLS, dtype=np.int64) // SIDE
_CELL_COL = np.arange(N_CELLS, dtype=np.int64) % SIDE


def _build_neighbors() -> np.ndarray:
    nbr = np.full((N_CELLS, 4), -1, dtype=np.int64)
    for c in range(N_CELLS):
        r, col = divmod(c, SIDE)
        cand = []
        if r > 0:
            cand.append(c - SIDE)
        if r < SIDE - 1:
            cand.append(c + SIDE)
        if col > 0:
            cand.append(c - 1)
        if col < SIDE - 1:
            cand.append(c + 1)
        nbr[c, : len(cand)] = cand
    return nbr


NEIGHBORS = _build_neighbors()   # [16, 4], -1 padded
_NEIGHBOR_LIST = [[int(c) for c in row if c >= 0] for row in NEIGHBORS]


def successors_scalar(state: int, blank: int) -> list[tuple[int, int]]:
    """[(child_state, child_blank)] via Python int bit-ops.

    Same transition as successors(); no numpy overhead, for sequential DFS
    hot paths (IDA*) where states are visited one at a time.
    """
    out = []
    bsh = blank * 4
    for n in _NEIGHBOR_LIST[blank]:
        nsh = n * 4
        t = (state >> nsh) & 0xF
        child = (state & ~(0xF << nsh) & ~(0xF << bsh)) | (t << bsh)
        out.append((child, n))
    return out


def as_state_array(states) -> np.ndarray:
    return np.asarray(states, dtype=np.uint64).reshape(-1)


def decode(states) -> np.ndarray:
    """uint64[N] -> uint8[N, 16], tiles[i, p] = tile at cell p."""
    s = as_state_array(states)
    return ((s[:, None] >> _SHIFTS) & np.uint64(0xF)).astype(np.uint8)


def encode(tiles) -> int:
    """[16] tile-per-cell -> packed state as a Python int."""
    s = 0
    for p, t in enumerate(tiles):
        s |= int(t) << (4 * p)
    return s


def find_blank(states) -> np.ndarray:
    """uint64[N] -> int64[N] cell index of the blank."""
    return np.argmax(decode(states) == 0, axis=1)


def successors(states, blanks=None):
    """Batch successor generation.

    Returns (succ uint64[N, 4], valid bool[N, 4], moved_to int64[N, 4]) where
    succ[i, j] is state i with the blank slid to cell moved_to[i, j]. Invalid
    slots (cells with fewer than 4 neighbors) have valid=False and succ equal
    to an arbitrary value that must be masked out.
    """
    s = as_state_array(states)
    if blanks is None:
        blanks = find_blank(s)
    nbr = NEIGHBORS[blanks]                                   # [N, 4]
    valid = nbr >= 0
    nbr_safe = np.where(valid, nbr, 0).astype(np.uint64)
    shift_n = nbr_safe * np.uint64(4)                          # [N, 4]
    shift_b = (blanks.astype(np.uint64) * np.uint64(4))[:, None]  # [N, 1]
    tile = (s[:, None] >> shift_n) & np.uint64(0xF)
    mask = ~((np.uint64(0xF) << shift_n) | (np.uint64(0xF) << shift_b))
    succ = (s[:, None] & mask) | (tile << shift_b)
    return succ, valid, nbr


def manhattan_sum(states) -> np.ndarray:
    """Standard admissible MD heuristic (blank excluded), int64[N]."""
    tiles = decode(states).astype(np.int64)
    d = np.abs(_CELL_ROW - tiles // SIDE) + np.abs(_CELL_COL - tiles % SIDE)
    d[tiles == 0] = 0
    return d.sum(axis=1)


def per_cell_md(states) -> np.ndarray:
    """float32[N, 16] per-cell MD including the blank cell.

    Matches phase2's PuzzleDatasetV2 x[256:272] feature exactly.
    """
    tiles = decode(states).astype(np.int64)
    d = np.abs(_CELL_ROW - tiles // SIDE) + np.abs(_CELL_COL - tiles % SIDE)
    return d.astype(np.float32)


def one_hot(states) -> np.ndarray:
    """float32[N, 256], index = cell*16 + tile. Matches phase2's PuzzleDataset."""
    tiles = decode(states).astype(np.int64)
    n = tiles.shape[0]
    x = np.zeros((n, 256), dtype=np.float32)
    x[np.arange(n)[:, None], _CELL_OFFSETS + tiles] = 1.0
    return x


def is_solvable(state: int) -> bool:
    """Permutation parity must match the blank's taxicab parity to cell 0."""
    tiles = decode([state])[0].astype(np.int64)
    inv = int(np.triu(tiles[:, None] > tiles[None, :], k=1).sum())
    blank = int(np.argmax(tiles == 0))
    taxicab = blank // SIDE + blank % SIDE
    return inv % 2 == taxicab % 2


def scramble(start: int, n_moves: int, rng: np.random.Generator) -> int:
    """Random walk of valid slides, never immediately undoing the last move."""
    s = int(start)
    blank = int(find_blank([s])[0])
    prev = -1
    for _ in range(n_moves):
        options = [c for c in NEIGHBORS[blank] if c >= 0 and c != prev]
        n = int(options[rng.integers(len(options))])
        succ, valid, moved = successors([s], np.array([blank]))
        j = int(np.argmax(moved[0] == n))
        s = int(succ[0, j])
        prev, blank = blank, n
    return s


def random_solvable_state(rng: np.random.Generator) -> int:
    """Uniform over the solvable half of the 16! permutations."""
    perm = rng.permutation(16)
    s = encode(perm)
    if is_solvable(s):
        return s
    a, b = (0, 1) if perm[0] != 0 and perm[1] != 0 else (2, 3)
    perm[a], perm[b] = perm[b], perm[a]
    return encode(perm)
