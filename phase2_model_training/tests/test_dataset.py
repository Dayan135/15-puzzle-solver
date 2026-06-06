#!/usr/bin/env python3
"""
test_dataset.py — validate PuzzleDataset, the cost-balanced sampler, and the
train/val/test split.

Two modes:
  1. Synthetic (default): build a tiny in-memory .bin with deliberate
     duplicates (a depth-1 state repeated 100k times) and assert the sampler
     flattens the cost distribution. Runs anywhere, no real data needed.
  2. Real: pass --data <path-or-dir> to additionally load the actual Phase 1
     dataset and print load time, dedup ratio, and a sampled-cost histogram.

Usage:
  python tests/test_dataset.py
  python tests/test_dataset.py --data data/full/
"""
import argparse
import struct
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# allow running from the phase2 dir: import data/puzzle_dataset.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.puzzle_dataset import (  # noqa: E402
    MAGIC,
    MAX_COST,
    PuzzleDataset,
    cost_balanced_sampler,
    train_val_test_split,
)

GOAL = 0xFEDCBA9876543210


def write_bin(path: Path, records):
    """records: list of (state_uint64, cost_uint8)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<IIQ", MAGIC, 1, len(records)))
        for s, c in records:
            f.write(struct.pack("<QB", s, c))


def slide(state, blank, nbr):
    """Move tile at cell `nbr` into the blank at `blank`; return (new_state, nbr)."""
    tile = (state >> (nbr * 4)) & 0xF
    state &= ~(0xF << (blank * 4))
    state |= tile << (blank * 4)
    state &= ~(0xF << (nbr * 4))
    return state, nbr


def make_synthetic(path: Path):
    """
    Build a dataset with extreme, known duplication:
      - one depth-1 state repeated 100_000 times  (cost 1)
      - 50_000 distinct-ish 'deep' states          (cost 50)
    The sampler must make cost 1 and cost 50 roughly equiprobable despite the
    100k:50k raw imbalance and the single distinct depth-1 state.
    """
    recs = []
    # depth-1: blank at cell 0 swaps with cell 1 -> a single distinct state
    d1, _ = slide(GOAL, 0, 1)
    recs += [(d1, 1)] * 100_000
    # 'deep' states: perturb high nibbles to fabricate 50k distinct uint64s,
    # all labelled cost 50 (label correctness is not under test here).
    rng = np.random.default_rng(0)
    seen = set()
    while len(seen) < 50_000:
        seen.add(int(rng.integers(0, 1 << 63, dtype=np.uint64)) | (1 << 63))
    recs += [(s, 50) for s in seen]
    write_bin(path, recs)
    return len(recs)


def test_synthetic():
    print("=== synthetic test ===")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "dataset_000.bin"
        n = make_synthetic(p)
        ds = PuzzleDataset(p, normalize_cost=True)
        assert len(ds) == n, f"len {len(ds)} != {n}"
        print(f"loaded {len(ds):,} records")

        # --- encoding checks ---
        x, y = ds[0]
        assert x.shape == (256,), x.shape
        assert x.sum().item() == 16.0, x.sum().item()         # one '1' per cell
        assert abs(y.item() - 1 / MAX_COST) < 1e-6, y.item()  # cost 1 normalised
        print(f"encoding OK: x.shape={tuple(x.shape)} x.sum={x.sum().item()} y={y.item():.5f}")

        # --- raw imbalance: cost 1 is 2x cost 50 by record count ---
        raw = Counter(ds.costs.tolist())
        print(f"raw counts: cost1={raw[1]:,}  cost50={raw[50]:,}  (ratio {raw[1]/raw[50]:.1f}x)")

        # --- sampler should flatten cost 1 vs cost 50 to ~1:1 ---
        sampler = cost_balanced_sampler(ds)
        loader = DataLoader(ds, batch_size=20_000, sampler=sampler)
        bx, by = next(iter(loader))
        costs = (by * MAX_COST).round().int().tolist()
        sc = Counter(costs)
        c1, c50 = sc.get(1, 0), sc.get(50, 0)
        ratio = c1 / c50 if c50 else float("inf")
        print(f"sampled (20k draws): cost1={c1:,}  cost50={c50:,}  (ratio {ratio:.2f}, want ~1.0)")
        assert 0.7 < ratio < 1.4, f"sampler did not balance: ratio={ratio:.2f}"
        print("sampler balances duplicated shallow vs diverse deep: OK")

        # --- split integrity ---
        tr, va, te = train_val_test_split(ds, seed=42)
        # val/test: one record per unique state, no duplicates within each
        tr_states = ds.states[np.array(tr.indices)]
        va_states = ds.states[np.array(va.indices)]
        te_states = ds.states[np.array(te.indices)]
        assert len(np.unique(va_states)) == len(va.indices), "val has duplicate states"
        assert len(np.unique(te_states)) == len(te.indices), "test has duplicate states"
        # zero state leakage between splits
        assert not np.isin(va_states, tr_states).any(), "val/train state overlap"
        assert not np.isin(te_states, tr_states).any(), "test/train state overlap"
        assert not np.isin(va_states, te_states).any(), "val/test state overlap"
        n_uniq = len(np.unique(ds.states))
        print(f"split OK: train={len(tr):,} val={len(va):,} test={len(te):,} "
              f"(unique states: {n_uniq:,}, no leakage)")
    print("synthetic test PASSED\n")


def test_real(path):
    print(f"=== real-data test: {path} ===")
    t0 = time.time()
    ds = PuzzleDataset(path, normalize_cost=True)
    dt = time.time() - t0
    n = len(ds)
    distinct = len(np.unique(ds.states))
    print(f"loaded {n:,} records in {dt:.1f}s")
    print(f"distinct states: {distinct:,}  ({100*distinct/n:.1f}% unique, "
          f"{n - distinct:,} duplicates)")

    raw = np.bincount(ds.costs.astype(np.int64), minlength=MAX_COST + 1)
    nz = [(c, int(raw[c])) for c in range(MAX_COST + 1) if raw[c]]
    print(f"raw cost range: {nz[0][0]}..{nz[-1][0]}  "
          f"(min-count={min(v for _, v in nz):,}  max-count={max(v for _, v in nz):,})")

    # sampled histogram: should be far flatter than raw
    sampler = cost_balanced_sampler(ds)
    loader = DataLoader(ds, batch_size=100_000, sampler=sampler, num_workers=2)
    by = next(iter(loader))[1]
    sc = Counter((by * MAX_COST).round().int().tolist())
    print("sampled cost histogram (100k draws):")
    mx = max(sc.values())
    for c in sorted(sc):
        bar = "#" * (sc[c] * 50 // mx)
        print(f"  {c:2d}: {bar} ({sc[c]})")
    print("real-data test done\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help=".bin file or directory of real Phase 1 data")
    args = ap.parse_args()

    test_synthetic()
    if args.data:
        test_real(args.data)
    print("ALL TESTS PASSED")
