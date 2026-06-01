import struct
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, WeightedRandomSampler

MAGIC    = 0x50313544          # "P15D"
MAX_COST = 80                  # theoretical 15-puzzle maximum
_HEADER  = struct.Struct("<IIQ")   # magic(4) + version(4) + n_records(8)
_RECORD  = struct.Struct("<QB")    # state(8) + cost(1)
_CHUNK   = 1 << 16                 # 65536 records per read pass


def _read_bin(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one .bin file. Returns (states uint64[N], costs uint8[N])."""
    with open(path, "rb") as f:
        magic, _ver, n = _HEADER.unpack(f.read(16))
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic 0x{magic:08X}")
        states = np.empty(n, dtype=np.uint64)
        costs  = np.empty(n, dtype=np.uint8)
        done   = 0
        while done < n:
            batch = min(_CHUNK, n - done)
            raw   = f.read(batch * 9)
            for i in range(batch):
                s, c = _RECORD.unpack_from(raw, i * 9)
                states[done + i] = s
                costs [done + i] = c
            done += batch
    return states, costs


class PuzzleDataset(Dataset):
    """
    Phase 1 binary dataset for supervised regression of h*(s).

    Args:
        paths:          .bin file, list of .bin files, or a directory
                        (all dataset_*.bin files in the directory are loaded).
        normalize_cost: if True, y = cost / MAX_COST (float in [0, 1]).
                        if False, y = float(cost).

    __getitem__ returns (x, y):
        x: float32 tensor [256] — one-hot per cell (16 cells × 16 tile values)
        y: float32 scalar       — normalised (or raw) optimal cost
    """

    def __init__(
        self,
        paths:          "str | Path | Sequence[str | Path]",
        normalize_cost: bool = True,
    ):
        paths = self._resolve_paths(paths)
        all_states, all_costs = zip(*(_read_bin(p) for p in paths))
        self.states         = np.concatenate(all_states)   # uint64[N]
        self.costs          = np.concatenate(all_costs)    # uint8[N]
        self.normalize_cost = normalize_cost

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        state = int(self.states[idx])
        cost  = int(self.costs[idx])

        # One-hot: 16 cells × 16 possible tiles = 256-dim float32.
        # Tile labels are categorical — feeding raw nibble values as floats
        # would impose an ordinal relationship (tile 12 > tile 3) that
        # doesn't exist in the puzzle.
        x = torch.zeros(256, dtype=torch.float32)
        for cell in range(16):
            tile = (state >> (cell * 4)) & 0xF
            x[cell * 16 + tile] = 1.0

        y = torch.tensor(
            cost / MAX_COST if self.normalize_cost else float(cost),
            dtype=torch.float32,
        )
        return x, y

    @staticmethod
    def _resolve_paths(paths) -> "list[Path]":
        if isinstance(paths, (str, Path)):
            p = Path(paths)
            return sorted(p.glob("dataset_*.bin")) if p.is_dir() else [p]
        return [Path(p) for p in paths]


def cost_balanced_sampler(dataset: PuzzleDataset) -> WeightedRandomSampler:
    """
    WeightedRandomSampler that gives each cost value equal expected frequency.

    Shallow buckets in Phase 1 are heavily duplicated (e.g. depth-1 has only
    2 distinct states but ~4.5M records). Rather than dedup-and-reload 900 MB,
    we assign weight = 1 / count(records at same cost) per record. A depth-1
    record with weight 1/4_500_000 is sampled as rarely as any single depth-52
    record, so duplicates have no effect on the effective training distribution.

    Usage:
        loader = DataLoader(ds, batch_size=B, sampler=cost_balanced_sampler(ds))
    Do NOT pass shuffle=True alongside a sampler.
    """
    costs  = dataset.costs.astype(np.int32)
    counts = np.bincount(costs, minlength=MAX_COST + 1).astype(np.float64)
    inv    = np.where(counts > 0, 1.0 / counts, 0.0)
    w      = torch.from_numpy(inv[costs]).float()
    return WeightedRandomSampler(w, num_samples=len(dataset), replacement=True)


def train_val_test_split(
    dataset:   PuzzleDataset,
    fractions: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed:      int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """
    Reproducible random split → (train, val, test) Subsets.
    All three share the underlying PuzzleDataset; no data is copied.
    """
    assert abs(sum(fractions) - 1.0) < 1e-6
    n    = len(dataset)
    idx  = np.random.default_rng(seed).permutation(n)
    n_tr = int(n * fractions[0])
    n_va = int(n * fractions[1])
    return (
        Subset(dataset, idx[:n_tr].tolist()),
        Subset(dataset, idx[n_tr:n_tr + n_va].tolist()),
        Subset(dataset, idx[n_tr + n_va:].tolist()),
    )
