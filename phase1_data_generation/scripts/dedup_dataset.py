#!/usr/bin/env python3
"""
dedup_dataset.py — collapse duplicate states in a 15-puzzle dataset.

Phase 1 generates stratified data: shallow scramble buckets contain only a
couple of distinct states but millions of records, so ~58% of the 100M records
are exact duplicates. Optimal cost is a deterministic function of the state, so
every duplicate of a state carries the identical label — they add zero
information and only cause train/val/test leakage and skewed metrics in Phase 2.

This script keeps one copy of each distinct state and writes a new dataset in
the same binary format. It is lossless (a sanity check asserts each state maps
to a single cost) and runs in-memory with one argsort over the records.

Binary format (phase1_data_generation):
  Header  16 B : magic(4) + version(4) + n_records(8)
  Record   9 B : state(uint64 LE) + cost(uint8)

IMPORTANT: write the output to a *different* directory than the input. Phase 2
globs dataset_*.bin, so a deduped file sitting next to the originals would be
loaded alongside them.

Usage:
  python scripts/dedup_dataset.py data/full --out-dir data/dedup
  python scripts/dedup_dataset.py data/full/dataset_000.bin --out-dir data/dedup
  python scripts/dedup_dataset.py data/full --out-dir data/dedup --on-conflict min
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC             = 0x50313544          # "P15D"
VERSION           = 1
HEADER            = struct.Struct("<IIQ")
REC_DTYPE         = np.dtype([("state", "<u8"), ("cost", "u1")])  # packed: itemsize 9
MAX_BYTES_DEFAULT = 1 << 30             # 1 GiB roll, matches Phase 1's split


def read_bin(path: Path):
    """Load one .bin file as a structured array; returns (records, version)."""
    with open(path, "rb") as f:
        magic, version, n = HEADER.unpack(f.read(16))
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic 0x{magic:08X} (expected 0x{MAGIC:08X})")
        arr = np.fromfile(f, dtype=REC_DTYPE, count=n)
    if len(arr) != n:
        raise ValueError(f"{path}: header claims {n} records, read {len(arr)}")
    return arr, version


def resolve_paths(src: Path):
    if src.is_dir():
        bins = sorted(src.glob("dataset_*.bin"))
        if not bins:
            sys.exit(f"No dataset_*.bin files found in {src}")
        return bins
    return [src]


def dedup(states: np.ndarray, costs: np.ndarray, on_conflict: str):
    """
    Return (uniq_states, uniq_costs) keeping one record per distinct state.

    Sort once by state; group boundaries are where the sorted state changes.
    reduceat over those boundaries gives each group's min/max cost in one pass,
    so a state appearing with two different costs (a Phase 1 bug) is detected.
    """
    order    = np.argsort(states, kind="stable")
    s_sorted = states[order]
    c_sorted = costs[order]

    first        = np.ones(len(s_sorted), dtype=bool)
    first[1:]    = s_sorted[1:] != s_sorted[:-1]
    starts       = np.nonzero(first)[0]
    uniq_states  = s_sorted[starts]

    gmin = np.minimum.reduceat(c_sorted, starts)
    gmax = np.maximum.reduceat(c_sorted, starts)

    conflict = gmin != gmax
    n_conflict = int(conflict.sum())
    if n_conflict:
        ex = np.nonzero(conflict)[0][:10]
        lines = "\n".join(
            f"    0x{int(uniq_states[i]):016X}: cost in [{int(gmin[i])}, {int(gmax[i])}]"
            for i in ex
        )
        msg = (f"{n_conflict:,} distinct state(s) carry more than one cost — "
               f"this indicates a Phase 1 labelling bug. First examples:\n{lines}")
        if on_conflict == "error":
            sys.exit("ERROR: " + msg + "\n(use --on-conflict min|max|first to override)")
        print("WARNING: " + msg, file=sys.stderr)

    if on_conflict == "min":
        uniq_costs = gmin
    elif on_conflict == "max":
        uniq_costs = gmax
    else:  # "first" (and the conflict-free "error" path)
        uniq_costs = c_sorted[starts]

    counts = np.diff(np.append(starts, len(s_sorted)))
    return uniq_states, uniq_costs, counts


def write_dataset(out_dir: Path, states: np.ndarray, costs: np.ndarray,
                  version: int, max_bytes: int):
    """Write records to dataset_NNN.bin, rolling at max_bytes like Phase 1."""
    out_dir.mkdir(parents=True, exist_ok=True)
    recs_per_file = (max_bytes - HEADER.size) // REC_DTYPE.itemsize

    out = np.empty(len(states), dtype=REC_DTYPE)
    out["state"] = states
    out["cost"]  = costs

    written = []
    idx = fno = 0
    while idx < len(out):
        chunk = out[idx: idx + recs_per_file]
        path  = out_dir / f"dataset_{fno:03d}.bin"
        with open(path, "wb") as f:
            f.write(HEADER.pack(MAGIC, version, len(chunk)))
            chunk.tofile(f)
        written.append((path, len(chunk)))
        idx += len(chunk)
        fno += 1
    return written


def main():
    ap = argparse.ArgumentParser(description="Deduplicate a 15-puzzle .bin dataset by state")
    ap.add_argument("src", help=".bin file or directory containing dataset_*.bin")
    ap.add_argument("--out-dir", required=True,
                    help="output directory (MUST differ from the input dir)")
    ap.add_argument("--on-conflict", choices=["error", "min", "max", "first"], default="error",
                    help="what to do if one state has multiple costs (default: error)")
    ap.add_argument("--max-bytes", type=int, default=MAX_BYTES_DEFAULT,
                    help=f"roll output files at this size (default: {MAX_BYTES_DEFAULT})")
    args = ap.parse_args()

    src      = Path(args.src)
    src_dir  = src if src.is_dir() else src.parent
    out_dir  = Path(args.out_dir)
    if out_dir.resolve() == src_dir.resolve():
        sys.exit(f"--out-dir must differ from the input dir ({src_dir}); "
                 f"Phase 2 would otherwise load originals + deduped together.")

    paths = resolve_paths(src)
    print(f"reading {len(paths)} file(s) from {src_dir} …", file=sys.stderr)
    arrs, version = [], VERSION
    for p in paths:
        a, version = read_bin(p)
        print(f"  {p.name}: {len(a):,} records", file=sys.stderr)
        arrs.append(a)
    arr = arrs[0] if len(arrs) == 1 else np.concatenate(arrs)

    n_total = len(arr)
    if n_total == 0:
        sys.exit("no records to deduplicate")

    uniq_states, uniq_costs, counts = dedup(arr["state"], arr["cost"], args.on_conflict)
    n_unique = len(uniq_states)
    n_dup    = n_total - n_unique

    written = write_dataset(out_dir, uniq_states, uniq_costs, version, args.max_bytes)

    print(f"\ntotal records : {n_total:,}", file=sys.stderr)
    print(f"distinct states: {n_unique:,}  ({100*n_unique/n_total:.1f}% unique)", file=sys.stderr)
    print(f"duplicates dropped: {n_dup:,}  ({100*n_dup/n_total:.1f}%)", file=sys.stderr)
    print(f"most-duplicated state appears {int(counts.max()):,} times", file=sys.stderr)
    for path, k in written:
        print(f"wrote {k:,} records -> {path}", file=sys.stderr)

    hist = np.bincount(uniq_costs.astype(np.int64))
    mx   = hist.max()
    print("\ndistinct states per cost (note: deep-skewed after dedup):", file=sys.stderr)
    for c in np.nonzero(hist)[0]:
        bar = "#" * (int(hist[c]) * 40 // int(mx))
        print(f"  {c:3d}: {bar} ({int(hist[c]):,})", file=sys.stderr)


if __name__ == "__main__":
    main()
