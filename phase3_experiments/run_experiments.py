"""Run a benchmark grid from a YAML config.

    python run_experiments.py --config configs/smoke_local.yaml

Config schema (paths are relative to the working directory; run from
phase3_experiments/):

    instances: testset/smoke.csv
    out: results/smoke.csv
    engine:
      batch_size: 128
      max_expansions: 200000
      max_time_s: 60
    algorithms:
      - {type: wastar, w: 1.0}
      - {type: gbfs}
    heuristics:
      - {type: md}
      - {type: classifier, checkpoint: ../phase2_model_training/checkpoints/classifier/best_model.pt,
         threshold: 0.5, temperature: 1.0}
"""

from __future__ import annotations

import argparse

import yaml

from search.algorithms import build_algorithm
from search.heuristics import build_heuristic
from benchmark.runner import load_instances, run_grid


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--no-resume", action="store_true",
                   help="overwrite the output CSV instead of skipping done runs")
    p.add_argument("--only-heuristic", type=int, default=None, metavar="I",
                   help="run only the I-th heuristic of the config "
                        "(cluster sharding: one slurm array task per heuristic)")
    p.add_argument("--out", default=None, help="override the config's out path")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.out:
        cfg["out"] = args.out

    heuristic_specs = cfg["heuristics"]
    if args.only_heuristic is not None:
        heuristic_specs = [heuristic_specs[args.only_heuristic]]

    instances = load_instances(cfg["instances"])
    algorithms = [build_algorithm(s) for s in cfg["algorithms"]]
    heuristics = [build_heuristic(s) for s in heuristic_specs]
    engine = cfg.get("engine", {})

    print(f"instances: {len(instances)}  algorithms: {len(algorithms)}  "
          f"heuristics: {len(heuristics)}")
    n = run_grid(
        instances, algorithms, heuristics, cfg["out"],
        batch_size=engine.get("batch_size", 64),
        max_expansions=engine.get("max_expansions"),
        max_time_s=engine.get("max_time_s"),
        resume=not args.no_resume,
    )
    print(f"done: {n} new rows -> {cfg['out']}")


if __name__ == "__main__":
    main()
