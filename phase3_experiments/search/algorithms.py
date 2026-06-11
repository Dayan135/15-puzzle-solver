"""Generic batched best-first search with uniform instrumentation.

One engine covers the whole algorithm family via the priority
    f(n) = g_weight * g(n) + h_weight * h(n)

    WA*(w):  g_weight=1, h_weight=w      (w=1 -> A*)
    GBFS:    g_weight=0, h_weight=1
    UCS:     g_weight=1, h_weight=0      (or ZeroHeuristic)

Every run returns the same SearchResult regardless of algorithm/heuristic, so
benchmarking (nodes expanded, heuristic time, wall time) is measured in exactly
one place.

batch_size > 1 expands the top-k of OPEN per iteration to amortize NN forward
passes. Solution quality is unaffected (the goal is only accepted when it pops
first), but nodes_expanded inflates by up to batch_size x for sharply guided
searches — a near-perfect heuristic still expands a full batch per depth
level. Use the default batch_size=1 whenever node counts are the metric;
batch_size is recorded with every result so mixed runs stay distinguishable.
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field

import numpy as np

from . import puzzle


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    g_weight: float
    h_weight: float
    params: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        suffix = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({suffix})" if suffix else self.name


def wastar(w: float) -> AlgorithmSpec:
    return AlgorithmSpec("wastar", g_weight=1.0, h_weight=float(w), params={"w": float(w)})


def gbfs() -> AlgorithmSpec:
    return AlgorithmSpec("gbfs", g_weight=0.0, h_weight=1.0)


def idastar(w: float = 1.0) -> AlgorithmSpec:
    return AlgorithmSpec("idastar", g_weight=1.0, h_weight=float(w), params={"w": float(w)})


def build_algorithm(spec: dict) -> AlgorithmSpec:
    spec = dict(spec)
    kind = spec.pop("type")
    if kind == "wastar":
        return wastar(spec.pop("w"))
    if kind == "gbfs":
        return gbfs()
    if kind == "idastar":
        return idastar(spec.pop("w", 1.0))
    raise ValueError(f"unknown algorithm type {kind!r}; known: wastar, gbfs, idastar")


@dataclass
class SearchResult:
    solved: bool
    termination: str            # solved | node_limit | time_limit | exhausted
    solution_cost: int | None
    nodes_expanded: int
    nodes_generated: int
    h_batches: int              # heuristic invocations (forward passes)
    h_states: int               # unique states evaluated by the heuristic
    h_time_s: float
    wall_time_s: float
    batch_size: int
    h_start: float = 0.0        # heuristic estimate at the start state
    path: list[int] | None = None


def best_first_search(
    start: int,
    heuristic,
    algorithm: AlgorithmSpec,
    *,
    batch_size: int = 1,
    max_expansions: int | None = None,
    max_time_s: float | None = None,
    track_path: bool = False,
) -> SearchResult:
    t_start = time.perf_counter()
    gw, hw = algorithm.g_weight, algorithm.h_weight

    stats = {"h_batches": 0, "h_states": 0, "h_time": 0.0}
    # h(goal) = 0 is a domain axiom, never a model query: run-2's classifier
    # predicts h(GOAL) = 58 (the goal is the only cost-0 state and the model
    # never saw it), which buries the goal node in the heap and makes every
    # search run to its node limit.
    h_cache: dict[int, float] = {puzzle.GOAL: 0.0}

    def ensure_h(states: list[int]) -> None:
        new = [s for s in dict.fromkeys(states) if s not in h_cache]
        if not new:
            return
        t0 = time.perf_counter()
        vals = heuristic(np.array(new, dtype=np.uint64))
        stats["h_time"] += time.perf_counter() - t0
        stats["h_batches"] += 1
        stats["h_states"] += len(new)
        h_cache.update(zip(new, map(float, vals)))

    def result(termination, goal_g, expanded, generated, parent):
        path = None
        if track_path and goal_g is not None:
            path, s = [], puzzle.GOAL
            while s is not None:
                path.append(s)
                s = parent[s]
            path.reverse()
        return SearchResult(
            solved=goal_g is not None,
            termination=termination,
            solution_cost=goal_g,
            nodes_expanded=expanded,
            nodes_generated=generated,
            h_batches=stats["h_batches"],
            h_states=stats["h_states"],
            h_time_s=stats["h_time"],
            wall_time_s=time.perf_counter() - t_start,
            batch_size=batch_size,
            h_start=h_start,
            path=path,
        )

    start = int(start)
    h_start = 0.0
    parent: dict[int, int | None] = {start: None} if track_path else None
    if start == puzzle.GOAL:
        return result("solved", 0, 0, 0, parent)

    ensure_h([start])
    h_start = h_cache[start]
    seq = itertools.count()
    # entries: (f, -g, seq, state) — ties prefer deeper nodes (standard WA*/GBFS)
    open_heap = [(gw * 0 + hw * h_cache[start], 0, next(seq), start)]
    best_g = {start: 0}
    nodes_expanded = 0
    nodes_generated = 0

    while open_heap:
        if max_time_s is not None and time.perf_counter() - t_start > max_time_s:
            return result("time_limit", None, nodes_expanded, nodes_generated, parent)

        cap = batch_size
        if max_expansions is not None:
            cap = min(cap, max_expansions - nodes_expanded)
            if cap <= 0:
                return result("node_limit", None, nodes_expanded, nodes_generated, parent)

        batch_states: list[int] = []
        batch_g: list[int] = []
        while open_heap and len(batch_states) < cap:
            f, neg_g, _, s = heapq.heappop(open_heap)
            g = -neg_g
            if g > best_g.get(s, g):
                continue                      # stale entry, a better g was pushed later
            if s == puzzle.GOAL:
                # Only terminate when the goal is the global min of OPEN.
                # Mid-batch, earlier pops with f <= f(goal) are still
                # unexpanded and may lead to a cheaper goal; push it back and
                # finish this batch first — next iteration pops it first.
                if not batch_states:
                    return result("solved", g, nodes_expanded, nodes_generated, parent)
                heapq.heappush(open_heap, (f, neg_g, next(seq), s))
                break
            batch_states.append(s)
            batch_g.append(g)
        if not batch_states:
            break

        nodes_expanded += len(batch_states)
        succ, valid, _ = puzzle.successors(np.array(batch_states, dtype=np.uint64))

        to_push: dict[int, tuple[int, int]] = {}   # state -> (g, parent)
        rows, cols = np.nonzero(valid)
        nodes_generated += len(rows)
        succ_flat = succ[rows, cols].tolist()
        for k in range(len(rows)):
            child = succ_flat[k]
            g_new = batch_g[rows[k]] + 1
            if g_new >= best_g.get(child, np.inf):
                continue
            prev = to_push.get(child)
            if prev is None or g_new < prev[0]:
                to_push[child] = (g_new, batch_states[rows[k]])

        if not to_push:
            continue
        children = list(to_push)
        ensure_h(children)
        for child in children:
            g_new, par = to_push[child]
            best_g[child] = g_new
            if track_path:
                parent[child] = par
            f = gw * g_new + hw * h_cache[child]
            heapq.heappush(open_heap, (f, -g_new, next(seq), child))

    return result("exhausted", None, nodes_expanded, nodes_generated, parent)


class _Limit(Exception):
    def __init__(self, kind: str):
        self.kind = kind


def ida_star_search(
    start: int,
    heuristic,
    algorithm: AlgorithmSpec | None = None,
    *,
    h_weight: float | None = None,
    max_expansions: int | None = None,
    max_time_s: float | None = None,
    track_path: bool = False,
    batch_size: int = 1,            # accepted for interface parity; DFS is sequential
) -> SearchResult:
    """IDA* (optionally weighted: f = g + w·h) with reverse-move pruning.

    Memory is O(depth) for the search itself; the h-cache grows with unique
    states visited, which keeps repeated iterations cheap. nodes_expanded is
    cumulative across iterations — the standard IDA* effort metric. With an
    admissible heuristic and w=1 the returned cost is optimal, so
    idastar + pdb doubles as the ground-truth oracle.
    """
    t_start = time.perf_counter()
    hw = h_weight if h_weight is not None else (algorithm.h_weight if algorithm else 1.0)

    stats = {"h_batches": 0, "h_states": 0, "h_time": 0.0}
    h_cache: dict[int, float] = {puzzle.GOAL: 0.0}
    nodes = {"expanded": 0, "generated": 0}

    def ensure_h(states: list[int]) -> None:
        new = [s for s in states if s not in h_cache]
        if not new:
            return
        t0 = time.perf_counter()
        vals = heuristic(np.array(new, dtype=np.uint64))
        stats["h_time"] += time.perf_counter() - t0
        stats["h_batches"] += 1
        stats["h_states"] += len(new)
        h_cache.update(zip(new, map(float, vals)))

    def result(termination, goal_g, path):
        return SearchResult(
            solved=goal_g is not None,
            termination=termination,
            solution_cost=goal_g,
            nodes_expanded=nodes["expanded"],
            nodes_generated=nodes["generated"],
            h_batches=stats["h_batches"],
            h_states=stats["h_states"],
            h_time_s=stats["h_time"],
            wall_time_s=time.perf_counter() - t_start,
            batch_size=1,
            h_start=h_start,
            path=path,
        )

    start = int(start)
    h_start = 0.0
    if start == puzzle.GOAL:
        return result("solved", 0, [start] if track_path else None)

    ensure_h([start])
    h_start = h_cache[start]
    start_blank = int(puzzle.find_blank([start])[0])
    FOUND = -1.0                    # safe sentinel: real f values are >= 0
    found = {"g": None}

    def dfs(s: int, g: int, blank: int, prev_blank: int, bound: float, path):
        f = g + hw * h_cache[s]
        if f > bound + 1e-9:
            return f
        if s == puzzle.GOAL:
            found["g"] = g
            return FOUND
        if max_expansions is not None and nodes["expanded"] >= max_expansions:
            raise _Limit("node_limit")
        if (max_time_s is not None and nodes["expanded"] % 1024 == 0
                and time.perf_counter() - t_start > max_time_s):
            raise _Limit("time_limit")

        nodes["expanded"] += 1
        children = [(c, nb) for c, nb in puzzle.successors_scalar(s, blank)
                    if nb != prev_blank]
        nodes["generated"] += len(children)
        ensure_h([c for c, _ in children])

        min_excess = float("inf")
        for child, nb in children:
            t = dfs(child, g + 1, nb, blank, bound, path)
            if t == FOUND:
                if path is not None:
                    path.append(child)
                return FOUND
            min_excess = min(min_excess, t)
        return min_excess

    bound = hw * h_start
    while True:
        path = [] if track_path else None
        try:
            t = dfs(start, 0, start_blank, -1, bound, path)
        except _Limit as e:
            return result(e.kind, None, None)
        if t == FOUND:
            if path is not None:
                path.append(start)
                path.reverse()
            return result("solved", found["g"], path)
        if t == float("inf"):
            return result("exhausted", None, None)
        bound = t


def run_search(start, heuristic, algorithm: AlgorithmSpec, **kw) -> SearchResult:
    """Dispatch to the right engine for the AlgorithmSpec."""
    if algorithm.name == "idastar":
        return ida_star_search(start, heuristic, algorithm, **kw)
    return best_first_search(start, heuristic, algorithm, **kw)
