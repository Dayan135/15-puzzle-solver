"""Suboptimality vs. search-effort Pareto figure from a summary CSV.

    python -m benchmark.plot_pareto results/cluster/walk1000_summary.csv
    python -m benchmark.plot_pareto results/cluster/walk1000_summary.csv \
        --out results/cluster/plots/walk1000_pareto.png

X axis: exp_avg (mean nodes expanded, log scale).
Y axis: subopt_avg (mean found/optimal; 1.0 = optimal).
Color = heuristic family, marker = algorithm. Pareto-optimal points
(no other point has both fewer expansions and lower suboptimality)
are circled and the frontier is drawn as a dashed step line.
"""

from __future__ import annotations

import argparse
import csv
import os
import re

import matplotlib.pyplot as plt

FAMILY_COLORS = {
    "pdb": "tab:blue",
    "regressor": "tab:orange",
    "classifier": "tab:green",
    "md": "tab:red",
    "zero": "tab:gray",
}
ALGO_MARKERS = {
    "idastar": "s",
    "wastar(w=1.0)": "o",
    "wastar(w=2.0)": "^",
    "wastar(w=5.0)": "v",
    "wastar(w=10.0)": "D",
    "gbfs": "X",
}


def family(heuristic: str) -> str:
    return heuristic.split("(")[0]


def algo_key(algorithm: str) -> str:
    if algorithm.startswith("idastar"):
        return "idastar"
    return algorithm


def short_label(heuristic: str) -> str:
    """classifier(temperature=2.0,threshold=0.2) -> clf t=0.2 T=2.0"""
    m = re.match(r"classifier\(temperature=([\d.]+),threshold=([\d.]+)\)", heuristic)
    if m:
        return f"clf t={m.group(2)} T={m.group(1)}"
    return heuristic


def pareto_front(points: list[dict]) -> list[dict]:
    """Points not dominated in (exp_avg, subopt_avg), both minimized."""
    front = []
    for p in points:
        if not any(q["exp_avg"] <= p["exp_avg"] and q["subopt_avg"] <= p["subopt_avg"]
                   and (q["exp_avg"] < p["exp_avg"] or q["subopt_avg"] < p["subopt_avg"])
                   for q in points):
            front.append(p)
    return sorted(front, key=lambda p: p["exp_avg"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary_csv")
    ap.add_argument("--out", default=None, help="output PNG (default: <csv_dir>/plots/<name>_pareto.png)")
    ap.add_argument("--annotate-front", action="store_true", default=True)
    args = ap.parse_args()

    with open(args.summary_csv, newline="") as f:
        rows = [{**r,
                 "exp_avg": float(r["exp_avg"]),
                 "subopt_avg": float(r["subopt_avg"])}
                for r in csv.DictReader(f)]

    front = pareto_front(rows)
    front_set = {(r["heuristic"], r["algorithm"]) for r in front}

    fig, ax = plt.subplots(figsize=(9, 6))
    seen_fam, seen_alg = set(), set()
    for r in rows:
        fam, alg = family(r["heuristic"]), algo_key(r["algorithm"])
        color = FAMILY_COLORS.get(fam, "tab:purple")
        marker = ALGO_MARKERS.get(alg, "*")
        ax.scatter(r["exp_avg"], r["subopt_avg"], c=color, marker=marker,
                   s=55, alpha=0.85, zorder=3,
                   edgecolors="black" if (r["heuristic"], r["algorithm"]) in front_set else "none",
                   linewidths=1.2)
        seen_fam.add(fam)
        seen_alg.add(alg)

    fx = [r["exp_avg"] for r in front]
    fy = [r["subopt_avg"] for r in front]
    ax.step(fx, fy, where="post", linestyle="--", color="black", alpha=0.5,
            zorder=2, label="Pareto frontier")
    if args.annotate_front:
        for r in front:
            ax.annotate(f'{short_label(r["heuristic"])}\n{r["algorithm"]}',
                        (r["exp_avg"], r["subopt_avg"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=7)

    ax.set_xscale("log")
    ax.set_xlabel("Mean nodes expanded (log)")
    ax.set_ylabel("Mean suboptimality (cost / optimal)")
    ax.set_title("15-Puzzle: solution quality vs. search effort (1000 uniform instances)")
    ax.axhline(1.0, color="gray", linewidth=0.6)
    ax.grid(True, which="both", alpha=0.25)

    fam_handles = [plt.Line2D([], [], marker="o", linestyle="", color=FAMILY_COLORS.get(f, "tab:purple"),
                              label=f) for f in sorted(seen_fam)]
    alg_handles = [plt.Line2D([], [], marker=ALGO_MARKERS.get(a, "*"), linestyle="",
                              color="black", label=a) for a in sorted(seen_alg)]
    leg1 = ax.legend(handles=fam_handles, title="heuristic", loc="upper right", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=alg_handles, title="algorithm", loc="center right", fontsize=8)

    out = args.out
    if out is None:
        base = os.path.splitext(os.path.basename(args.summary_csv))[0]
        out = os.path.join(os.path.dirname(args.summary_csv), "plots", f"{base}_pareto.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")

    print("\nPareto-optimal configurations:")
    for r in front:
        print(f'  {r["heuristic"]:42s} {r["algorithm"]:15s} '
              f'subopt={r["subopt_avg"]:.3f} exp={r["exp_avg"]:.1f}')


if __name__ == "__main__":
    main()
