"""
Run-3 dataset: 256 one-hot + 16 per-cell Manhattan distances = 272-dim input.

Returns (x [272], y, md_sum) per item where:
  x[0:256]   one-hot encoding identical to PuzzleDataset
  x[256:272] Manhattan distance of each cell's tile to its goal, normalised by 6 to [0, 1] (includes blank cell)
  y          h*(s) — raw optimal cost (always; residual is y - md_sum)
  md_sum     Σ MD for the 15 non-blank tiles = standard admissible Manhattan heuristic

The x[256:272] blank-distance feature tells the model where the blank sits
geometrically. The md_sum excludes the blank, matching the standard heuristic
convention.  Training loop computes residual = y - md_sum when needed.
"""
import time
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from .puzzle_dataset import (
    MAX_COST, _SHIFTS, _CELL_OFFSETS, _read_bin,
    CostBalancedSampler, train_val_test_split,
)

INPUT_DIM = 272

# Module-level constants reused in every __getitem__ call.
_CELL_ROW    = np.arange(16, dtype=np.int32) // 4   # row index of each cell (0–3)
_CELL_COL    = np.arange(16, dtype=np.int32) % 4    # col index of each cell (0–3)
_MAX_CELL_MD = 6.0   # max per-cell Manhattan distance in a 4×4 grid (corner to corner)


class PuzzleDatasetV2(Dataset):
    """
    272-dim input variant of PuzzleDataset.

    __getitem__ → (x: float32[272], y: float32, md_sum: float32)

    Per-cell Manhattan distances (x[256:272]) are computed on-the-fly in
    __getitem__ to keep memory usage comparable to the original dataset.
    md_sums ([N] float32) are pre-computed at init (~400 MB for 100M records).
    """

    def __init__(
        self,
        paths: "str | Path | Sequence[str | Path]",
        normalize_cost: bool = False,
    ):
        paths = self._resolve_paths(paths)
        print(f"  reading {len(paths)} file(s)…", flush=True)
        t0 = time.time()
        all_states, all_costs = zip(*(_read_bin(p) for p in paths))
        self.states         = np.concatenate(all_states)   # uint64[N]
        self.costs          = np.concatenate(all_costs)    # uint8[N]
        self.normalize_cost = normalize_cost
        print(f"  {len(self.states):,} records loaded in {time.time()-t0:.1f}s", flush=True)

        # Decode nibbles and compute MD sums in chunks to cap peak RAM.
        #
        # Naive full-array approach instantiates [N,16] uint64 (12.8 GB) for
        # nibbles and four simultaneous [N,16] int32 arrays (~25 GB) for the MD
        # computation — totalling ~40 GB, which OOMs a 32 GB node.
        # Chunked processing keeps peak at ~5 GB regardless of dataset size.
        N      = len(self.states)
        CHUNK  = 5_000_000    # 5 M records → ~640 MB uint64 temp per nibble chunk
        t1     = time.time()
        print("  decoding nibbles + MD sums (chunked)…", flush=True)
        self.nibbles = np.empty((N, 16), dtype=np.uint8)
        self.md_sums = np.empty(N,        dtype=np.float32)

        for start in range(0, N, CHUNK):
            end   = min(start + CHUNK, N)
            chunk = self.states[start:end]

            # Nibble decode: temp [chunk, 16] uint64 (~640 MB), freed after cast
            nib = ((chunk[:, None] >> _SHIFTS) & np.uint64(0xF)).astype(np.uint8)
            self.nibbles[start:end] = nib

            # MD sums: four [chunk, 16] int32 arrays (~64 MB each), freed per chunk
            t   = nib.astype(np.int32)
            gr  = t // 4
            gc  = t % 4
            d   = np.abs(_CELL_ROW - gr) + np.abs(_CELL_COL - gc)
            d[t == 0] = 0
            self.md_sums[start:end] = d.sum(axis=1).astype(np.float32)

        print(f"  nibbles + MD sums ready in {time.time()-t1:.1f}s", flush=True)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        tiles    = self.nibbles[idx].astype(np.int32)                    # [16]
        goal_row = tiles // 4
        goal_col = tiles % 4
        cell_dists = (
            np.abs(_CELL_ROW - goal_row) + np.abs(_CELL_COL - goal_col)
        ).astype(np.float32)                                             # [16], includes blank

        x = np.zeros(272, dtype=np.float32)
        x[_CELL_OFFSETS + self.nibbles[idx]] = 1.0           # one-hot [0:256]
        x[256:272] = cell_dists / _MAX_CELL_MD               # MD features [256:272], normalised to [0, 1]

        cost = float(self.costs[idx])
        y    = cost / MAX_COST if self.normalize_cost else cost
        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(self.md_sums[idx], dtype=torch.float32),
        )

    @staticmethod
    def _resolve_paths(paths) -> "list[Path]":
        if isinstance(paths, (str, Path)):
            p = Path(paths)
            return sorted(p.glob("dataset_*.bin")) if p.is_dir() else [p]
        return [Path(p) for p in paths]


__all__ = ["INPUT_DIM", "PuzzleDatasetV2", "CostBalancedSampler", "train_val_test_split"]
