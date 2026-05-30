#!/usr/bin/env python3
"""
bin_to_csv.py — convert a 15-puzzle dataset_NNN.bin to CSV.

Binary format (phase1_data_generation):
  Header  16 B : magic(4) + version(4) + n_records(8)
  Record   9 B : state(uint64 LE) + cost(uint8)

CSV columns:
  state_hex, t0..t15  (tile at each cell, 0=blank), cost

Usage:
  python scripts/bin_to_csv.py data/full/dataset_000.bin
  python scripts/bin_to_csv.py data/full/dataset_000.bin -o out.csv
  python scripts/bin_to_csv.py data/full/dataset_000.bin --limit 100000
  python scripts/bin_to_csv.py data/full/              # all .bin files in dir
"""

import argparse
import csv
import struct
import sys
from pathlib import Path

MAGIC   = 0x50313544  # "P15D"
CHUNK   = 65536       # records per read batch

HEADER  = "state_hex," + ",".join(f"t{i}" for i in range(16)) + ",cost\n"


def decode_state(s: int):
    return [(s >> (i * 4)) & 0xF for i in range(16)]


def iter_records(path: Path, limit=None):
    with open(path, "rb") as f:
        magic, version = struct.unpack_from("<II", f.read(8))
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic 0x{magic:08X} (expected 0x{MAGIC:08X})")
        (n_records,) = struct.unpack_from("<Q", f.read(8))

        total = min(n_records, limit) if limit else n_records
        yielded = 0
        fmt = f"<{CHUNK}QB"  # won't work for mixed types — use per-record unpack
        rec_fmt = struct.Struct("<QB")

        while yielded < total:
            batch = min(CHUNK, total - yielded)
            raw = f.read(batch * 9)
            if not raw:
                break
            for i in range(batch):
                state, cost = rec_fmt.unpack_from(raw, i * 9)
                yield state, cost
            yielded += batch


def convert(src: Path, dst: Path, limit, quiet: bool) -> int:
    count = 0
    with open(dst, "w", newline="") as out:
        out.write(HEADER)
        writer = csv.writer(out)
        for state, cost in iter_records(src, limit):
            tiles = decode_state(state)
            writer.writerow([f"0x{state:016X}"] + tiles + [cost])
            count += 1
            if not quiet and count % 500_000 == 0:
                print(f"  {count:,} records written...", file=sys.stderr)
    return count


def main():
    ap = argparse.ArgumentParser(description="Convert .bin dataset to CSV")
    ap.add_argument("src", help=".bin file or directory containing .bin files")
    ap.add_argument("-o", "--out", default=None,
                    help="output CSV path (default: same name as src with .csv)")
    ap.add_argument("--limit", type=int, default=None,
                    help="max records to convert (default: all)")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="suppress progress output")
    args = ap.parse_args()

    src = Path(args.src)
    if src.is_dir():
        bins = sorted(src.glob("dataset_*.bin"))
        if not bins:
            sys.exit(f"No dataset_*.bin files found in {src}")
        # merge into one CSV
        dst = Path(args.out) if args.out else src / "dataset.csv"
        total = 0
        with open(dst, "w", newline="") as out:
            out.write(HEADER)
            writer = csv.writer(out)
            for b in bins:
                if not args.quiet:
                    print(f"Reading {b.name}...", file=sys.stderr)
                remaining = (args.limit - total) if args.limit else None
                for state, cost in iter_records(b, remaining):
                    tiles = decode_state(state)
                    writer.writerow([f"0x{state:016X}"] + tiles + [cost])
                    total += 1
                    if not args.quiet and total % 500_000 == 0:
                        print(f"  {total:,} records written...", file=sys.stderr)
                    if args.limit and total >= args.limit:
                        break
                if args.limit and total >= args.limit:
                    break
        count = total
    else:
        dst = Path(args.out) if args.out else src.with_suffix(".csv")
        if not args.quiet:
            print(f"Reading {src.name}...", file=sys.stderr)
        count = convert(src, dst, args.limit, args.quiet)

    print(f"Wrote {count:,} records -> {dst}", file=sys.stderr)

    # quick cost histogram
    if not args.quiet:
        import collections
        costs = collections.Counter()
        with open(dst, newline="") as f:
            next(f)  # skip header
            for line in f:
                costs[int(line.rstrip().split(",")[-1])] += 1
        print("\nCost histogram:", file=sys.stderr)
        for c in sorted(costs):
            bar = "#" * (costs[c] * 40 // max(costs.values()))
            print(f"  {c:3d}: {bar} ({costs[c]:,})", file=sys.stderr)


if __name__ == "__main__":
    main()
