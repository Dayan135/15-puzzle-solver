from collections import deque

import numpy as np
import pytest

from search import puzzle
from search.algorithms import (
    best_first_search,
    build_algorithm,
    gbfs,
    ida_star_search,
    idastar,
    run_search,
    wastar,
)
from search.heuristics import ManhattanHeuristic, ZeroHeuristic


def bfs_optimal_cost(start: int, limit: int = 200_000) -> int:
    """Reference breadth-first search for ground-truth optimal cost."""
    if start == puzzle.GOAL:
        return 0
    seen = {start}
    frontier = deque([(start, 0)])
    while frontier:
        s, g = frontier.popleft()
        succ, valid, _ = puzzle.successors([s])
        for j in np.nonzero(valid[0])[0]:
            c = int(succ[0, j])
            if c == puzzle.GOAL:
                return g + 1
            if c not in seen:
                seen.add(c)
                frontier.append((c, g + 1))
        if len(seen) > limit:
            raise RuntimeError("BFS limit hit")
    raise RuntimeError("unsolvable")


def scrambles(n, moves, seed):
    rng = np.random.default_rng(seed)
    return [puzzle.scramble(puzzle.GOAL, moves, rng) for _ in range(n)]


def test_start_is_goal():
    r = best_first_search(puzzle.GOAL, ManhattanHeuristic(), wastar(1.0))
    assert r.solved and r.solution_cost == 0 and r.nodes_expanded == 0


def test_astar_md_is_optimal():
    for s in scrambles(8, 14, seed=10):
        opt = bfs_optimal_cost(s)
        r = best_first_search(s, ManhattanHeuristic(), wastar(1.0), track_path=True)
        assert r.solved and r.termination == "solved"
        assert r.solution_cost == opt
        # path is a valid move sequence from start to goal
        assert r.path[0] == s and r.path[-1] == puzzle.GOAL
        assert len(r.path) == r.solution_cost + 1
        for a, b in zip(r.path, r.path[1:]):
            succ, valid, _ = puzzle.successors([a])
            assert b in {int(x) for x in succ[0][valid[0]]}


def test_ucs_matches_bfs():
    for s in scrambles(4, 10, seed=11):
        r = best_first_search(s, ZeroHeuristic(), wastar(1.0))
        assert r.solution_cost == bfs_optimal_cost(s)


def test_batch_size_invariant_optimality():
    for s in scrambles(4, 12, seed=12):
        costs = {
            best_first_search(s, ManhattanHeuristic(), wastar(1.0), batch_size=b).solution_cost
            for b in (1, 4, 64)
        }
        assert len(costs) == 1


def test_batch_size_invariant_optimality_deep():
    # Regression: with a large batch, a goal popped mid-batch used to be
    # accepted even though smaller-f nodes in the same batch were unexpanded,
    # returning suboptimal solutions at w=1. Only shows up on deeper instances.
    for s in scrambles(3, 40, seed=19):
        ref = best_first_search(s, ManhattanHeuristic(), wastar(1.0), batch_size=1)
        big = best_first_search(s, ManhattanHeuristic(), wastar(1.0), batch_size=128)
        assert big.solution_cost == ref.solution_cost


def test_gbfs_solves_with_bounded_quality():
    for s in scrambles(5, 14, seed=13):
        opt = bfs_optimal_cost(s)
        r = best_first_search(s, ManhattanHeuristic(), gbfs(), track_path=True)
        assert r.solved
        assert r.solution_cost >= opt
        assert len(r.path) == r.solution_cost + 1


def test_weighted_astar_bounded_suboptimality():
    w = 2.0
    for s in scrambles(5, 12, seed=14):
        opt = bfs_optimal_cost(s)
        r = best_first_search(s, ManhattanHeuristic(), wastar(w))
        assert r.solved
        assert opt <= r.solution_cost <= w * opt


def test_node_limit_termination():
    s = scrambles(1, 300, seed=15)[0]
    r = best_first_search(s, ManhattanHeuristic(), wastar(1.0), max_expansions=50)
    assert not r.solved
    assert r.termination == "node_limit"
    assert r.nodes_expanded <= 50


def test_time_limit_termination():
    s = scrambles(1, 300, seed=16)[0]
    r = best_first_search(s, ManhattanHeuristic(), wastar(1.0), max_time_s=0.05)
    assert not r.solved
    assert r.termination == "time_limit"


def test_instrumentation_populated():
    s = scrambles(1, 20, seed=17)[0]
    r = best_first_search(s, ManhattanHeuristic(), gbfs(), batch_size=16)
    assert r.nodes_expanded > 0
    assert r.nodes_generated >= r.nodes_expanded
    assert r.h_states > 0
    assert r.h_batches > 0
    assert r.h_time_s >= 0
    assert r.wall_time_s >= r.h_time_s
    assert r.batch_size == 16


def test_goal_h_is_axiomatically_zero():
    # A heuristic that misjudges the goal (run-2 classifier predicts h(GOAL)=58)
    # must not prevent termination: the engine pins h(goal) = 0.
    class GoalBlindMD(ManhattanHeuristic):
        def __call__(self, states):
            h = super().__call__(states)
            h[np.asarray(states, dtype=np.uint64) == np.uint64(puzzle.GOAL)] = 58.0
            return h

    for s in scrambles(3, 12, seed=18):
        r = best_first_search(s, GoalBlindMD(), gbfs(), max_expansions=50_000)
        assert r.solved


def test_build_algorithm_specs():
    a = build_algorithm({"type": "wastar", "w": 1.5})
    assert a.g_weight == 1.0 and a.h_weight == 1.5 and a.params == {"w": 1.5}
    g = build_algorithm({"type": "gbfs"})
    assert g.g_weight == 0.0 and g.h_weight == 1.0
    i = build_algorithm({"type": "idastar"})
    assert i.name == "idastar" and i.h_weight == 1.0
    iw = build_algorithm({"type": "idastar", "w": 1.5})
    assert iw.h_weight == 1.5
    with pytest.raises(ValueError):
        build_algorithm({"type": "dfs"})


def test_idastar_md_is_optimal():
    for s in scrambles(6, 14, seed=21):
        opt = bfs_optimal_cost(s)
        r = ida_star_search(s, ManhattanHeuristic(), idastar(), track_path=True)
        assert r.solved and r.termination == "solved"
        assert r.solution_cost == opt
        assert r.path[0] == s and r.path[-1] == puzzle.GOAL
        assert len(r.path) == r.solution_cost + 1
        for a, b in zip(r.path, r.path[1:]):
            assert b in {c for c, _ in puzzle.successors_scalar(
                a, int(puzzle.find_blank([a])[0]))}


def test_idastar_matches_astar_on_deeper_states():
    for s in scrambles(3, 40, seed=22):
        a = best_first_search(s, ManhattanHeuristic(), wastar(1.0))
        i = ida_star_search(s, ManhattanHeuristic(), idastar())
        assert i.solution_cost == a.solution_cost


def test_idastar_goal_start_and_limits():
    assert ida_star_search(puzzle.GOAL, ManhattanHeuristic(), idastar()).solution_cost == 0
    s = scrambles(1, 300, seed=23)[0]
    r = ida_star_search(s, ManhattanHeuristic(), idastar(), max_expansions=100)
    assert not r.solved and r.termination == "node_limit"
    assert r.nodes_expanded <= 101


def test_idastar_instrumentation_and_h_start():
    s = scrambles(1, 20, seed=24)[0]
    r = ida_star_search(s, ManhattanHeuristic(), idastar())
    assert r.h_start == float(puzzle.manhattan_sum([s])[0])
    assert r.nodes_expanded > 0 and r.nodes_generated > 0
    assert r.h_states > 0 and r.h_batches > 0
    assert r.wall_time_s >= r.h_time_s >= 0


def test_weighted_idastar_bounded():
    w = 2.0
    for s in scrambles(4, 12, seed=25):
        opt = bfs_optimal_cost(s)
        r = ida_star_search(s, ManhattanHeuristic(), idastar(w))
        assert r.solved
        assert opt <= r.solution_cost <= w * opt


def test_run_search_dispatch():
    s = scrambles(1, 10, seed=26)[0]
    ri = run_search(s, ManhattanHeuristic(), idastar())
    rb = run_search(s, ManhattanHeuristic(), wastar(1.0))
    assert ri.solution_cost == rb.solution_cost


def test_successors_scalar_matches_vectorized():
    rng = np.random.default_rng(27)
    for s in [puzzle.scramble(puzzle.GOAL, 50, rng) for _ in range(20)]:
        blank = int(puzzle.find_blank([s])[0])
        scalar = set(puzzle.successors_scalar(s, blank))
        succ, valid, moved = puzzle.successors([s])
        vec = {(int(succ[0, j]), int(moved[0, j]))
               for j in np.nonzero(valid[0])[0]}
        assert scalar == vec
