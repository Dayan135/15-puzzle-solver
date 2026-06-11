"""Benchmark grid driver: algorithms x heuristics x instances -> CSV.

Each row is one (instance, algorithm, heuristic) run with the full uniform
metric set from SearchResult. Rows are flushed as they complete and existing
(instance, algorithm, heuristic) keys are skipped on rerun, so an interrupted
cluster job resumes by resubmitting with the same output file.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

from search.algorithms import AlgorithmSpec, run_search
from search.heuristics import Heuristic

CSV_FIELDS = [
    "instance_id", "state_hex", "optimal_cost",
    "algorithm", "algo_params", "heuristic", "heuristic_params", "h_start",
    "solved", "termination", "solution_cost", "suboptimality",
    "nodes_expanded", "nodes_generated",
    "h_batches", "h_states", "h_time_s", "wall_time_s", "batch_size",
]


@dataclass(frozen=True)
class Instance:
    id: int
    state: int
    optimal_cost: int | None


def load_instances(path) -> list[Instance]:
    """CSV with header: id,state_hex,optimal_cost (optimal_cost may be empty)."""
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            opt = row.get("optimal_cost", "")
            out.append(Instance(
                id=int(row["id"]),
                state=int(row["state_hex"], 16),
                optimal_cost=int(opt) if opt not in ("", None) else None,
            ))
    return out


def save_instances(instances, path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "state_hex", "optimal_cost"])
        for inst in instances:
            w.writerow([inst.id, f"{inst.state:016x}",
                        "" if inst.optimal_cost is None else inst.optimal_cost])


def _run_key(instance_id, algorithm, algo_params, heuristic, heuristic_params):
    return (str(instance_id), algorithm, algo_params, heuristic, heuristic_params)


def _existing_keys(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        header = next(csv.reader(f), None)
        if header != CSV_FIELDS:
            raise ValueError(
                f"{path} has a different column schema; rerun with resume=False "
                f"(--no-resume) or move the old file away"
            )
        f.seek(0)
        return {
            _run_key(r["instance_id"], r["algorithm"], r["algo_params"],
                     r["heuristic"], r["heuristic_params"])
            for r in csv.DictReader(f)
        }


def run_grid(
    instances: list[Instance],
    algorithms: list[AlgorithmSpec],
    heuristics: list[Heuristic],
    out_csv,
    *,
    batch_size: int = 1,
    max_expansions: int | None = None,
    max_time_s: float | None = None,
    resume: bool = True,
    log=print,
) -> int:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done = _existing_keys(out_csv) if resume else set()
    write_header = not out_csv.exists() or not resume

    n_written = 0
    total = len(instances) * len(algorithms) * len(heuristics)
    mode = "w" if write_header else "a"
    with open(out_csv, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for heuristic in heuristics:
            h_params = json.dumps(heuristic.params(), sort_keys=True)
            for algo in algorithms:
                a_params = json.dumps(algo.params, sort_keys=True)
                t0 = time.perf_counter()
                n_solved = n_run = 0
                for inst in instances:
                    key = _run_key(inst.id, algo.name, a_params, heuristic.name, h_params)
                    if key in done:
                        continue
                    r = run_search(
                        inst.state, heuristic, algo,
                        batch_size=batch_size,
                        max_expansions=max_expansions,
                        max_time_s=max_time_s,
                    )
                    subopt = ""
                    if r.solved and inst.optimal_cost:
                        subopt = f"{r.solution_cost / inst.optimal_cost:.4f}"
                    writer.writerow({
                        "instance_id": inst.id,
                        "state_hex": f"{inst.state:016x}",
                        "optimal_cost": "" if inst.optimal_cost is None else inst.optimal_cost,
                        "algorithm": algo.name,
                        "algo_params": a_params,
                        "heuristic": heuristic.name,
                        "heuristic_params": h_params,
                        "h_start": f"{r.h_start:.2f}",
                        "solved": int(r.solved),
                        "termination": r.termination,
                        "solution_cost": "" if r.solution_cost is None else r.solution_cost,
                        "suboptimality": subopt,
                        "nodes_expanded": r.nodes_expanded,
                        "nodes_generated": r.nodes_generated,
                        "h_batches": r.h_batches,
                        "h_states": r.h_states,
                        "h_time_s": f"{r.h_time_s:.4f}",
                        "wall_time_s": f"{r.wall_time_s:.4f}",
                        "batch_size": r.batch_size,
                    })
                    f.flush()
                    n_written += 1
                    n_run += 1
                    n_solved += r.solved
                log(f"[grid] {heuristic.name} x {algo.label}: "
                    f"{n_solved}/{n_run} solved in {time.perf_counter() - t0:.1f}s "
                    f"({n_written}/{total} rows total)")
    return n_written
