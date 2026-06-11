"""Generate benchmark instances (without ground-truth optimal costs).

Optimal costs come later from phase1's IDA*+PDB oracle on the cluster; this
generator covers smoke tests and relative comparisons in the meantime.

    python -m benchmark.gen_instances --n 50 --mode scramble --moves 30 \
        --seed 0 --out testset/smoke.csv
"""

from __future__ import annotations

import argparse

import numpy as np

from search import puzzle
from benchmark.runner import Instance, save_instances


def generate(n: int, mode: str, moves: int, seed: int) -> list[Instance]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        if mode == "uniform":
            s = puzzle.random_solvable_state(rng)
        elif mode == "scramble":
            s = puzzle.scramble(puzzle.GOAL, moves, rng)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        out.append(Instance(id=i, state=s, optimal_cost=None))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--mode", choices=["uniform", "scramble"], default="uniform")
    p.add_argument("--moves", type=int, default=1000, help="scramble length (scramble mode)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    instances = generate(args.n, args.mode, args.moves, args.seed)
    save_instances(instances, args.out)
    print(f"wrote {len(instances)} instances to {args.out}")


if __name__ == "__main__":
    main()
