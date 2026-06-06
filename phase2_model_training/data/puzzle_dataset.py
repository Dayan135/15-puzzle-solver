import struct
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, Subset

MAGIC    = 0x50313544          # "P15D"
MAX_COST = 80                  # theoretical 15-puzzle maximum
_HEADER  = struct.Struct("<IIQ")   # magic(4) + version(4) + n_records(8)

# Module-level constants reused in every __getitem__ call (allocated once).
_SHIFTS       = np.arange(16, dtype=np.uint64) * np.uint64(4)  # bit-shift amounts per cell
_CELL_OFFSETS = np.arange(16, dtype=np.int32)  * 16             # one-hot base index per cell


def _read_bin(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one .bin file. Returns (states uint64[N], costs uint8[N]).

    Reads the entire file in one syscall and extracts fields with numpy,
    avoiding a Python loop over every record.
    """
    with open(path, "rb") as f:
        magic, _ver, n = _HEADER.unpack(f.read(16))
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic 0x{magic:08X}")
        raw = np.frombuffer(f.read(n * 9), dtype=np.uint8).reshape(n, 9)
    # Records are packed (no padding): 8 bytes state (LE uint64) + 1 byte cost.
    states = np.frombuffer(np.ascontiguousarray(raw[:, :8]).tobytes(), dtype="<u8").copy()
    costs  = np.ascontiguousarray(raw[:, 8])
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
        # Pre-decode nibbles once: [N, 16] uint8.  Eliminates bit-ops inside
        # __getitem__, which is called ~78M times per epoch by DataLoader workers.
        # Memory cost: N×16 bytes (~1.6 GB for 100M records, well within 32 GB).
        self.nibbles = (
            (self.states[:, None] >> _SHIFTS) & np.uint64(0xF)
        ).astype(np.uint8)   # [N, 16] uint8

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        # One-hot: 16 cells × 16 possible tiles = 256-dim float32.
        # Tile labels are categorical — ordinal encoding would impose a false
        # ordering (tile 12 > tile 3) that doesn't exist in the puzzle.
        x = np.zeros(256, dtype=np.float32)
        x[_CELL_OFFSETS + self.nibbles[idx]] = 1.0   # vectorised; no Python loop

        cost = float(self.costs[idx])
        y    = cost / MAX_COST if self.normalize_cost else cost
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

    @staticmethod
    def _resolve_paths(paths) -> "list[Path]":
        if isinstance(paths, (str, Path)):
            p = Path(paths)
            return sorted(p.glob("dataset_*.bin")) if p.is_dir() else [p]
        return [Path(p) for p in paths]


class CostBalancedSampler(Sampler):
    """
    Sampler that draws each cost value with equal expected frequency: pick a
    present cost uniformly at random, then a record with that cost uniformly.

    This is mathematically identical to weighting each record by
    1 / count(records at same cost), but avoids torch's WeightedRandomSampler,
    whose multinomial backend caps at 2**24 (~16.7M) categories — our dataset
    has 100M records, which overflows that cap.

    Shallow buckets in Phase 1 are heavily duplicated (e.g. depth-1 has only a
    couple of distinct states but ~4.5M records); grouping by cost makes those
    duplicates irrelevant to the effective training distribution at zero memory
    cost (only per-cost index lists are stored).

    Usage:
        loader = DataLoader(ds, batch_size=B, sampler=CostBalancedSampler(ds))
    Do NOT pass shuffle=True alongside a sampler.
    """

    def __init__(self, dataset: PuzzleDataset, num_samples: int = None, seed: int = None):
        costs = dataset.costs.astype(np.int64)
        # Group dataset indices by cost, stored as one flat array + per-group
        # (start, size) so sampling is fully vectorised (no Python per-item loop).
        order = np.argsort(costs, kind="stable")          # indices sorted by cost
        sorted_costs = costs[order]
        bounds = np.searchsorted(sorted_costs, np.arange(MAX_COST + 2))
        present = [c for c in range(MAX_COST + 1) if bounds[c + 1] > bounds[c]]
        self._flat   = order
        self._starts = np.array([bounds[c]     for c in present], dtype=np.int64)
        self._sizes  = np.array([bounds[c + 1] - bounds[c] for c in present], dtype=np.int64)
        self._num_samples = num_samples if num_samples is not None else len(dataset)
        self._seed = seed

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self):
        rng = np.random.default_rng(self._seed)
        n_groups = len(self._sizes)
        g       = rng.integers(0, n_groups, size=self._num_samples)        # uniform over costs
        within  = (rng.random(self._num_samples) * self._sizes[g]).astype(np.int64)
        picks   = self._flat[self._starts[g] + within]
        yield from picks.tolist()


def cost_balanced_sampler(dataset: PuzzleDataset, **kwargs) -> CostBalancedSampler:
    """Convenience factory; see CostBalancedSampler."""
    return CostBalancedSampler(dataset, **kwargs)


def train_val_test_split(
    dataset:   PuzzleDataset,
    fractions: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed:      int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """
    Dedup-aware split → (train, val, test) Subsets.

    Fractions apply to *unique states*, not records:
      - val/test: exactly one record per unique state (first occurrence)
      - train:    ALL records whose state falls in the train bucket,
                  including duplicates (CostBalancedSampler handles them)

    This prevents data leakage: a state seen during training never appears
    in val or test, so metrics reflect generalisation, not memorisation.
    """
    assert abs(sum(fractions) - 1.0) < 1e-6

    states = dataset.states                                   # uint64[N]
    unique_states, first_idx = np.unique(states, return_index=True)
    n_uniq = len(unique_states)

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(n_uniq)
    n_va = int(n_uniq * fractions[1])
    n_te = int(n_uniq * fractions[2])

    val_perm   = perm[:n_va]
    test_perm  = perm[n_va:n_va + n_te]
    train_perm = perm[n_va + n_te:]

    # val/test: one index (first occurrence) per unique state
    val_idx  = first_idx[val_perm]
    test_idx = first_idx[test_perm]

    # train: every record whose state maps to a train-bucket unique state
    is_train_uniq = np.zeros(n_uniq, dtype=bool)
    is_train_uniq[train_perm] = True
    record_ranks = np.searchsorted(unique_states, states)     # O(N log n_uniq)
    train_idx = np.where(is_train_uniq[record_ranks])[0]

    # Pass numpy arrays directly — Subset accepts any sequence and numpy int64
    # works as an index into PuzzleDataset.  Avoids creating three Python lists
    # totalling ~80M elements (which would allocate ~2 GB of extra memory).
    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )
