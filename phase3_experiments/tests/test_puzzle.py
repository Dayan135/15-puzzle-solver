import numpy as np
import pytest

from search import puzzle


def slide_ref(state: int, blank: int, n: int) -> int:
    """Scalar reference port of phase1's slide()."""
    t = (state >> (n * 4)) & 0xF
    state &= ~(0xF << (blank * 4))
    state |= t << (blank * 4)
    state &= ~(0xF << (n * 4))
    return state


def manhattan_ref(state: int) -> int:
    total = 0
    for p in range(16):
        t = (state >> (p * 4)) & 0xF
        if t == 0:
            continue
        total += abs(p // 4 - t // 4) + abs(p % 4 - t % 4)
    return total


def test_goal_decode():
    tiles = puzzle.decode([puzzle.GOAL])[0]
    assert list(tiles) == list(range(16))


def test_encode_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        perm = rng.permutation(16)
        s = puzzle.encode(perm)
        assert list(puzzle.decode([s])[0]) == list(perm)


def test_goal_successors():
    succ, valid, moved = puzzle.successors([puzzle.GOAL])
    assert valid[0].sum() == 2                       # blank at corner cell 0
    assert sorted(moved[0][valid[0]].tolist()) == [1, 4]
    for j in np.nonzero(valid[0])[0]:
        expected = slide_ref(puzzle.GOAL, 0, int(moved[0, j]))
        assert int(succ[0, j]) == expected


def test_successors_match_scalar_reference():
    rng = np.random.default_rng(1)
    states = [puzzle.scramble(puzzle.GOAL, 80, rng) for _ in range(50)]
    succ, valid, moved = puzzle.successors(np.array(states, dtype=np.uint64))
    blanks = puzzle.find_blank(states)
    for i, s in enumerate(states):
        for j in range(4):
            if not valid[i, j]:
                continue
            assert int(succ[i, j]) == slide_ref(s, int(blanks[i]), int(moved[i, j]))


def test_manhattan_matches_reference():
    rng = np.random.default_rng(2)
    states = [puzzle.GOAL] + [puzzle.scramble(puzzle.GOAL, k, rng) for k in (1, 5, 40, 200)]
    md = puzzle.manhattan_sum(states)
    assert md[0] == 0
    for i, s in enumerate(states):
        assert md[i] == manhattan_ref(s)


def test_per_cell_md_consistent_with_sum():
    rng = np.random.default_rng(3)
    states = [puzzle.scramble(puzzle.GOAL, 100, rng) for _ in range(10)]
    per_cell = puzzle.per_cell_md(states)
    blanks = puzzle.find_blank(states)
    md = puzzle.manhattan_sum(states)
    for i in range(len(states)):
        blank_d = per_cell[i, blanks[i]]
        assert per_cell[i].sum() - blank_d == md[i]


def test_one_hot_layout():
    x = puzzle.one_hot([puzzle.GOAL])
    assert x.shape == (1, 256) and x.dtype == np.float32
    assert x.sum() == 16
    expected = np.zeros(256, dtype=np.float32)
    expected[[p * 16 + p for p in range(16)]] = 1.0   # goal: cell p holds tile p
    assert np.array_equal(x[0], expected)


def test_solvability():
    assert puzzle.is_solvable(puzzle.GOAL)
    rng = np.random.default_rng(4)
    for k in (1, 2, 17, 100):
        assert puzzle.is_solvable(puzzle.scramble(puzzle.GOAL, k, rng))
    # swapping two non-blank tiles flips solvability
    tiles = list(range(16))
    tiles[1], tiles[2] = tiles[2], tiles[1]
    assert not puzzle.is_solvable(puzzle.encode(tiles))


def test_random_solvable_state_uniformity_of_parity():
    rng = np.random.default_rng(5)
    for _ in range(50):
        assert puzzle.is_solvable(puzzle.random_solvable_state(rng))


def test_scramble_never_unsolvable_and_changes_state():
    rng = np.random.default_rng(6)
    s = puzzle.scramble(puzzle.GOAL, 1000, rng)
    assert s != puzzle.GOAL
    assert puzzle.is_solvable(s)
