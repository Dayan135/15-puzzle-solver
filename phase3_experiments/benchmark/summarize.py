"""Aggregate a benchmark CSV into the heuristic-quality / search-effort table.

    python -m benchmark.summarize results/smoke.csv
    python -m benchmark.summarize results/smoke.csv --csv results/smoke_summary.csv

Per (heuristic, algorithm) group:
    real_avg   mean true cost over instances
    h_avg      mean h(start) over instances
    diff_std   std  of h(start) - real
    diff_max   max  of h(start) - real   (worst overestimation; negative = always under)
    over_rate  fraction of instances with h(start) > real (admissibility violations)
    cost_avg   mean found solution cost (solved runs)
    subopt_avg mean of found/real per solved instance
    exp_avg    mean nodes expanded
    gen_avg    mean nodes generated
    time_avg   mean wall seconds per instance
    h_frac     fraction of total wall time spent in the heuristic

True cost per instance comes from the optimal_cost column when present;
otherwise it falls back to the best solution found by any run of that instance
in the file — exact whenever the grid contains an admissible w=1 config
(e.g. md + wastar(w=1.0)), an upper bound otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict

_NOISE_PARAMS = {"checkpoint", "device", "pdb_dir"}   # paths/runtime, not identity

SUMMARY_FIELDS = ["heuristic", "algorithm", "n", "solved",
                  "real_avg", "h_avg", "diff_std", "diff_max", "over_rate",
                  "cost_avg", "subopt_avg",
                  "exp_avg", "gen_avg", "time_avg", "h_frac"]


def heuristic_label(row: dict) -> str:
    params = {k: v for k, v in json.loads(row["heuristic_params"]).items()
              if k not in _NOISE_PARAMS}
    suffix = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"{row['heuristic']}({suffix})" if suffix else row["heuristic"]


def algorithm_label(row: dict) -> str:
    params = json.loads(row["algo_params"])
    suffix = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"{row['algorithm']}({suffix})" if suffix else row["algorithm"]


def true_costs(rows: list[dict]) -> tuple[dict, bool]:
    """instance_id -> true cost. Second value: True if any cost is a fallback."""
    costs: dict[str, int] = {}
    fallback = False
    best: dict[str, int] = defaultdict(lambda: 10**9)
    for r in rows:
        if r["optimal_cost"]:
            costs[r["instance_id"]] = int(r["optimal_cost"])
        elif r["solved"] == "1" and r["solution_cost"]:
            best[r["instance_id"]] = min(best[r["instance_id"]], int(r["solution_cost"]))
    for iid, c in best.items():
        if iid not in costs:
            costs[iid] = c
            fallback = True
    return costs, fallback


def summarize(rows: list[dict]) -> tuple[list[dict], bool]:
    real, fallback = true_costs(rows)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(heuristic_label(r), algorithm_label(r))].append(r)

    out = []
    for (h, alg), grp in sorted(groups.items()):
        diffs = [float(r["h_start"]) - real[r["instance_id"]]
                 for r in grp if r["instance_id"] in real]
        reals = [real[r["instance_id"]] for r in grp if r["instance_id"] in real]
        costs = [int(r["solution_cost"]) for r in grp if r["solved"] == "1"]
        subopts = [int(r["solution_cost"]) / real[r["instance_id"]]
                   for r in grp
                   if r["solved"] == "1" and real.get(r["instance_id"])]
        wall = sum(float(r["wall_time_s"]) for r in grp)
        h_time = sum(float(r["h_time_s"]) for r in grp)
        out.append({
            "heuristic": h,
            "algorithm": alg,
            "n": len(grp),
            "solved": sum(r["solved"] == "1" for r in grp),
            "real_avg": statistics.mean(reals) if reals else float("nan"),
            "h_avg": statistics.mean(float(r["h_start"]) for r in grp),
            "diff_std": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
            "diff_max": max(diffs) if diffs else float("nan"),
            "over_rate": (statistics.mean(d > 0 for d in diffs)
                          if diffs else float("nan")),
            "cost_avg": statistics.mean(costs) if costs else float("nan"),
            "subopt_avg": statistics.mean(subopts) if subopts else float("nan"),
            "exp_avg": statistics.mean(int(r["nodes_expanded"]) for r in grp),
            "gen_avg": statistics.mean(int(r["nodes_generated"]) for r in grp),
            "time_avg": wall / len(grp),
            "h_frac": h_time / wall if wall else 0.0,
        })
    return out, fallback


_COL_FORMATS = {                      # numeric columns: (width, format)
    "n": (4, "d"), "solved": (6, "d"),
    "real_avg": (8, ".2f"), "h_avg": (7, ".2f"),
    "diff_std": (8, ".2f"), "diff_max": (8, ".1f"), "over_rate": (9, ".0%"),
    "cost_avg": (8, ".2f"), "subopt_avg": (10, ".3f"),
    "exp_avg": (9, ".1f"), "gen_avg": (9, ".1f"),
    "time_avg": (8, ".3f"), "h_frac": (6, ".0%"),
}


def format_table(summary: list[dict]) -> str:
    widths = {"heuristic": max([len("heuristic")] + [len(s["heuristic"]) for s in summary]),
              "algorithm": max([len("algorithm")] + [len(s["algorithm"]) for s in summary])}
    header = (f"{'heuristic':{widths['heuristic']}s}  {'algorithm':{widths['algorithm']}s}"
              + "".join(f" {name:>{w}s}" for name, (w, _) in _COL_FORMATS.items()))
    lines = [header, "-" * len(header)]
    for s in summary:
        row = f"{s['heuristic']:{widths['heuristic']}s}  {s['algorithm']:{widths['algorithm']}s}"
        for name, (w, fmt) in _COL_FORMATS.items():
            row += f" {s[name]:{w}{fmt}}"
        lines.append(row)
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_csv", nargs="+",
                   help="one or more per-run CSVs (shards are merged)")
    p.add_argument("--csv", help="also write the summary as CSV")
    args = p.parse_args()

    rows = []
    for path in args.results_csv:
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        raise SystemExit("no rows in input files")

    summary, fallback = summarize(rows)
    if fallback:
        print("note: optimal_cost missing for some instances — using the best "
              "solution found in this file (exact only if an admissible w=1 "
              "config is present)\n")
    print(format_table(summary))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            w.writeheader()
            w.writerows(summary)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
