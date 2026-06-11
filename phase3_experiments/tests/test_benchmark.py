import csv
import statistics

import numpy as np
import pytest

from search import puzzle
from search.algorithms import gbfs, wastar
from search.heuristics import ManhattanHeuristic
from benchmark.runner import Instance, load_instances, run_grid, save_instances
from benchmark.summarize import summarize


def make_instances(n=4, moves=12, seed=20):
    rng = np.random.default_rng(seed)
    return [Instance(id=i, state=puzzle.scramble(puzzle.GOAL, moves, rng),
                     optimal_cost=None) for i in range(n)]


def test_instances_roundtrip(tmp_path):
    instances = make_instances()
    path = tmp_path / "inst.csv"
    save_instances(instances, path)
    assert load_instances(path) == instances


def test_run_grid_writes_h_start(tmp_path):
    instances = make_instances()
    out = tmp_path / "res.csv"
    run_grid(instances, [wastar(1.0)], [ManhattanHeuristic()], out, log=lambda *_: None)
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == len(instances)
    by_id = {int(r["instance_id"]): r for r in rows}
    for inst in instances:
        md = float(puzzle.manhattan_sum([inst.state])[0])
        assert float(by_id[inst.id]["h_start"]) == md


def test_run_grid_resume_skips_done(tmp_path):
    instances = make_instances()
    out = tmp_path / "res.csv"
    n1 = run_grid(instances, [wastar(1.0)], [ManhattanHeuristic()], out, log=lambda *_: None)
    n2 = run_grid(instances, [wastar(1.0)], [ManhattanHeuristic()], out, log=lambda *_: None)
    assert n1 == len(instances) and n2 == 0
    assert len(list(csv.DictReader(open(out)))) == len(instances)


def test_resume_rejects_schema_mismatch(tmp_path):
    out = tmp_path / "res.csv"
    out.write_text("instance_id,old_column\n0,1\n")
    with pytest.raises(ValueError, match="schema"):
        run_grid(make_instances(), [wastar(1.0)], [ManhattanHeuristic()], out,
                 log=lambda *_: None)


def test_summarize_metrics(tmp_path):
    instances = make_instances()
    out = tmp_path / "res.csv"
    # wastar(1.0)+md is admissible & optimal -> its solution costs are the
    # ground-truth fallback; gbfs rows share the same true costs.
    run_grid(instances, [wastar(1.0), gbfs()], [ManhattanHeuristic()], out,
             log=lambda *_: None)
    rows = list(csv.DictReader(open(out)))
    summary, fallback = summarize(rows)
    assert fallback                      # no optimal_cost column values
    assert len(summary) == 2             # one group per algorithm

    astar_rows = [r for r in rows if r["algorithm"] == "wastar"]
    real = {r["instance_id"]: int(r["solution_cost"]) for r in astar_rows}
    diffs = [float(r["h_start"]) - real[r["instance_id"]] for r in astar_rows]

    g = next(s for s in summary if s["algorithm"].startswith("wastar"))
    assert g["n"] == g["solved"] == len(instances)
    assert g["real_avg"] == pytest.approx(statistics.mean(real.values()))
    assert g["h_avg"] == pytest.approx(statistics.mean(
        float(r["h_start"]) for r in astar_rows))
    assert g["diff_std"] == pytest.approx(statistics.stdev(diffs))
    assert g["diff_max"] == pytest.approx(max(diffs))
    assert g["exp_avg"] == pytest.approx(statistics.mean(
        int(r["nodes_expanded"]) for r in astar_rows))
    assert g["gen_avg"] == pytest.approx(statistics.mean(
        int(r["nodes_generated"]) for r in astar_rows))
    assert g["cost_avg"] == pytest.approx(statistics.mean(real.values()))
    assert g["subopt_avg"] == pytest.approx(1.0)   # admissible w=1 is optimal
    assert g["time_avg"] > 0
    assert 0.0 <= g["h_frac"] <= 1.0
    # MD is admissible: h never exceeds the true cost
    assert g["diff_max"] <= 0
    assert g["over_rate"] == 0.0


def test_summarize_prefers_optimal_cost_column(tmp_path):
    instances = [Instance(id=i, state=s.state, optimal_cost=99)
                 for i, s in enumerate(make_instances(n=2))]
    out = tmp_path / "res.csv"
    run_grid(instances, [wastar(1.0)], [ManhattanHeuristic()], out, log=lambda *_: None)
    summary, fallback = summarize(list(csv.DictReader(open(out))))
    assert not fallback
    assert summary[0]["real_avg"] == 99
