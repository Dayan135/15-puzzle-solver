import struct
from pathlib import Path

import numpy as np
import pytest

from search import puzzle
from search.heuristics import (
    ManhattanHeuristic,
    PDBHeuristic,
    ZeroHeuristic,
    build_heuristic,
)

PHASE2 = Path(__file__).resolve().parents[2] / "phase2_model_training"
CLASSIFIER_CKPT = PHASE2 / "checkpoints/classifier/best_model.pt"
REGRESSOR_CKPT = PHASE2 / "checkpoints/regressor/best_model.pt"
REAL_PDB_DIR = Path(__file__).resolve().parents[2] / "phase1_data_generation/data/pdbs"


def rank_ref(state: int, group) -> int:
    """Scalar port of phase1's AdditivePDB::rank (pdb.cpp)."""
    pos = [0] * 16
    for p in range(16):
        pos[(state >> (p * 4)) & 0xF] = p
    used, r = 0, 0
    for i, tile in enumerate(group):
        cell = pos[tile]
        smaller = cell - bin(used & ((1 << cell) - 1)).count("1")
        r = r * (16 - i) + smaller
        used |= 1 << cell
    return r


def write_pdb(path, group, table):
    with open(path, "wb") as f:
        f.write(struct.pack("<II", 0x50444231, len(group)))
        for t in group:
            f.write(struct.pack("<i", t))
        f.write(struct.pack("<Q", len(table)))
        f.write(bytes(table))


def pdb_size(k):
    n = 1
    for i in range(k):
        n *= 16 - i
    return n


def test_md_heuristic_matches_manhattan():
    rng = np.random.default_rng(0)
    states = np.array([puzzle.scramble(puzzle.GOAL, 60, rng) for _ in range(20)],
                      dtype=np.uint64)
    h = ManhattanHeuristic()(states)
    assert h.dtype == np.float32
    assert np.array_equal(h, puzzle.manhattan_sum(states).astype(np.float32))


def test_zero_heuristic():
    h = ZeroHeuristic()(np.array([puzzle.GOAL], dtype=np.uint64))
    assert h.tolist() == [0.0]


def test_pdb_heuristic_matches_reference_rank(tmp_path):
    rng = np.random.default_rng(1)
    group_a, group_b = [1, 2], [3, 4, 5]
    table_a = bytes(rng.integers(0, 30, pdb_size(2), dtype=np.uint8))
    table_b = bytes(rng.integers(0, 30, pdb_size(3), dtype=np.uint8))
    write_pdb(tmp_path / "pdb_a.bin", group_a, table_a)
    write_pdb(tmp_path / "pdb_b.bin", group_b, table_b)

    h = PDBHeuristic(tmp_path)
    assert h._group_a == group_a and h._group_b == group_b

    states = [puzzle.GOAL] + [puzzle.scramble(puzzle.GOAL, 100, rng) for _ in range(30)]
    got = h(np.array(states, dtype=np.uint64))
    for i, s in enumerate(states):
        expected = table_a[rank_ref(s, group_a)] + table_b[rank_ref(s, group_b)]
        assert got[i] == expected


def test_pdb_rejects_bad_magic(tmp_path):
    (tmp_path / "pdb_a.bin").write_bytes(b"\x00" * 32)
    (tmp_path / "pdb_b.bin").write_bytes(b"\x00" * 32)
    with pytest.raises(ValueError):
        PDBHeuristic(tmp_path)


def test_build_heuristic_registry():
    assert isinstance(build_heuristic({"type": "md"}), ManhattanHeuristic)
    with pytest.raises(ValueError):
        build_heuristic({"type": "nope"})


needs_ckpt = pytest.mark.skipif(
    not CLASSIFIER_CKPT.exists(), reason="run-2 checkpoints not present"
)


@pytest.mark.skipif(not (REAL_PDB_DIR / "pdb_b.bin").exists(),
                    reason="real 7-8 PDB tables not present")
def test_real_pdb_tables():
    h = PDBHeuristic(REAL_PDB_DIR)
    assert h._group_a == [1, 2, 3, 4, 5, 6, 7]
    assert h._group_b == [8, 9, 10, 11, 12, 13, 14, 15]

    rng = np.random.default_rng(7)
    moves_list = [0, 1, 6, 12, 25, 60, 200, 1000]
    states = np.array([puzzle.scramble(puzzle.GOAL, m, rng) for m in moves_list],
                      dtype=np.uint64)
    vals = h(states)
    assert vals[0] == 0                                   # h(GOAL) = 0
    # additive 7-8 PDB dominates Manhattan distance
    assert np.all(vals >= puzzle.manhattan_sum(states))
    # admissible: scramble length upper-bounds the optimal cost
    assert np.all(vals <= np.array(moves_list))


@needs_ckpt
def test_classifier_heuristic_sane():
    rng = np.random.default_rng(2)
    states = np.array(
        [puzzle.GOAL] + [puzzle.scramble(puzzle.GOAL, k, rng) for k in (5, 30, 200, 1000)],
        dtype=np.uint64,
    )
    h = build_heuristic({
        "type": "classifier", "checkpoint": str(CLASSIFIER_CKPT),
        "threshold": 0.5, "temperature": 1.0,
    })
    vals = h(states)
    assert vals.shape == (5,) and vals.dtype == np.float32
    assert np.all((vals >= 0) & (vals <= puzzle.MAX_COST))
    assert np.all(vals == np.round(vals))          # integer-valued cost classes

    # lower CDF threshold can only lower the predicted quantile
    lo = build_heuristic({
        "type": "classifier", "checkpoint": str(CLASSIFIER_CKPT),
        "threshold": 0.05, "temperature": 1.0,
    })(states)
    assert np.all(lo <= vals)


@needs_ckpt
def test_regressor_heuristic_sane():
    rng = np.random.default_rng(3)
    states = np.array(
        [puzzle.GOAL] + [puzzle.scramble(puzzle.GOAL, k, rng) for k in (5, 30, 200)],
        dtype=np.uint64,
    )
    h = build_heuristic({"type": "regressor", "checkpoint": str(REGRESSOR_CKPT)})
    vals = h(states)
    assert vals.shape == (4,) and vals.dtype == np.float32
    assert np.all((vals >= 0) & (vals <= puzzle.MAX_COST))
    assert np.all(vals == np.round(vals))


@needs_ckpt
def test_nn_heuristic_roughly_tracks_difficulty():
    rng = np.random.default_rng(4)
    easy = np.array([puzzle.scramble(puzzle.GOAL, 4, rng) for _ in range(32)],
                    dtype=np.uint64)
    hard = np.array([puzzle.scramble(puzzle.GOAL, 1000, rng) for _ in range(32)],
                    dtype=np.uint64)
    h = build_heuristic({"type": "classifier", "checkpoint": str(CLASSIFIER_CKPT)})
    assert h(easy).mean() + 10 < h(hard).mean()
