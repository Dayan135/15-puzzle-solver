"""Heuristic adapters with one uniform batch interface.

Every heuristic is a callable `h(states: uint64[N]) -> float32[N]` plus
`name` / `params()` metadata for benchmark records. The search engine never
knows which kind it is talking to — PDB, Manhattan, or a neural net.

Torch is imported lazily so MD/PDB-only runs work without it. Registry-based
construction (`build_heuristic`) keeps the experiment grid declarative: a YAML
spec like `{type: classifier, checkpoint: ..., threshold: 0.1}` maps directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from . import puzzle

_PHASE2_DIR = Path(__file__).resolve().parents[2] / "phase2_model_training"

_POPCOUNT16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)


class Heuristic:
    """Batch heuristic: __call__(uint64[N]) -> float32[N]."""

    name: str = "base"
    is_admissible: bool = False

    def params(self) -> dict:
        return {}

    def __call__(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ZeroHeuristic(Heuristic):
    """h = 0 — turns WA* into uniform-cost search. Baseline / debugging."""

    name = "zero"
    is_admissible = True

    def __call__(self, states):
        return np.zeros(len(states), dtype=np.float32)


class ManhattanHeuristic(Heuristic):
    name = "md"
    is_admissible = True

    def __call__(self, states):
        return puzzle.manhattan_sum(states).astype(np.float32)


class PDBHeuristic(Heuristic):
    """Additive 7-8 PDB lookup, mirroring phase1's AdditivePDB::rank exactly.

    Reads the PDB1 binary format written by phase1's build_pdbs.
    """

    name = "pdb"
    is_admissible = True

    def __init__(self, pdb_dir, file_a: str = "pdb_a.bin", file_b: str = "pdb_b.bin"):
        self.pdb_dir = str(pdb_dir)
        self._group_a, self._table_a = self._load(Path(pdb_dir) / file_a)
        self._group_b, self._table_b = self._load(Path(pdb_dir) / file_b)

    @staticmethod
    def _load(path: Path):
        with open(path, "rb") as f:
            header = f.read(8)
            magic = int.from_bytes(header[0:4], "little")
            k = int.from_bytes(header[4:8], "little")
            if magic != 0x50444231 or not (0 < k <= puzzle.N_CELLS):
                raise ValueError(f"{path}: not a PDB1 file")
            group = [int.from_bytes(f.read(4), "little", signed=True) for _ in range(k)]
            n = int.from_bytes(f.read(8), "little")
            table = np.fromfile(f, dtype=np.uint8, count=n)
            if len(table) != n:
                raise ValueError(f"{path}: truncated table ({len(table)} of {n})")
        return group, table

    @staticmethod
    def _rank(tiles: np.ndarray, group) -> np.ndarray:
        """Partial-permutation Lehmer rank, vectorized over states."""
        n = tiles.shape[0]
        pos = np.empty((n, puzzle.N_CELLS), dtype=np.int64)
        pos[np.arange(n)[:, None], tiles.astype(np.int64)] = np.arange(puzzle.N_CELLS)
        used = np.zeros(n, dtype=np.int64)
        r = np.zeros(n, dtype=np.uint64)
        for i, tile in enumerate(group):
            cell = pos[:, tile]
            below = used & ((1 << cell) - 1)
            smaller = cell - _POPCOUNT16[below]
            r = r * np.uint64(puzzle.N_CELLS - i) + smaller.astype(np.uint64)
            used |= 1 << cell
        return r

    def __call__(self, states):
        tiles = puzzle.decode(states)
        ha = self._table_a[self._rank(tiles, self._group_a)]
        hb = self._table_b[self._rank(tiles, self._group_b)]
        return (ha.astype(np.float32) + hb.astype(np.float32))

    def params(self):
        return {"pdb_dir": self.pdb_dir}


class _TorchHeuristic(Heuristic):
    """Shared loading/batching/device logic for all NN adapters."""

    def __init__(self, checkpoint, device: str | None = None, eval_batch_size: int = 8192):
        import torch

        self._torch = torch
        self.checkpoint = str(checkpoint)
        self.eval_batch_size = eval_batch_size
        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = device
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = ckpt.get("cfg", {})
        self.model = self._build_model(self.cfg)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval().to(device)

    @staticmethod
    def _phase2_models():
        if str(_PHASE2_DIR) not in sys.path:
            sys.path.insert(0, str(_PHASE2_DIR))
        import model as phase2_models

        return phase2_models

    def _build_model(self, cfg: dict):
        raise NotImplementedError

    def _encode(self, states) -> np.ndarray:
        raise NotImplementedError

    def _predict(self, x):  # torch [B, D] -> torch [B] heuristic values
        raise NotImplementedError

    def __call__(self, states):
        torch = self._torch
        x_np = self._encode(states)
        out = np.empty(len(x_np), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(x_np), self.eval_batch_size):
                xb = torch.from_numpy(x_np[i : i + self.eval_batch_size]).to(self.device)
                hb = self._predict(xb)
                out[i : i + len(xb)] = hb.float().cpu().numpy()
        return out

    def params(self):
        return {"checkpoint": self.checkpoint, "device": self.device}


class ClassifierHeuristic(_TorchHeuristic):
    """Run-2 classifier: h = smallest k with P(cost <= k | logits/T) >= threshold.

    threshold=0.5, temperature=1.0 reproduces predict_quantile's median; the
    run-2 sweep's >=99%-admissible point is threshold=0.10, temperature=2.0.
    """

    name = "classifier"

    def __init__(self, checkpoint, threshold: float = 0.5, temperature: float = 1.0, **kw):
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        super().__init__(checkpoint, **kw)

    def _build_model(self, cfg):
        m = self._phase2_models()
        return m.PuzzleClassifier(width=cfg.get("width", 256), depth=cfg.get("depth", 4))

    def _encode(self, states):
        return puzzle.one_hot(states)

    def _predict(self, x):
        logits = self.model(x) / self.temperature
        cdf = self._torch.softmax(logits, dim=-1).cumsum(dim=-1)
        return (cdf < self.threshold).sum(dim=-1).clamp(0, puzzle.MAX_COST)

    def params(self):
        return {**super().params(), "threshold": self.threshold, "temperature": self.temperature}


class RegressorHeuristic(_TorchHeuristic):
    """Run-2 pinball regressor: h = round(clamp(output, 0, 80))."""

    name = "regressor"

    def _build_model(self, cfg):
        m = self._phase2_models()
        return m.PuzzleRegressor(width=cfg.get("width", 256), depth=cfg.get("depth", 4))

    def _encode(self, states):
        return puzzle.one_hot(states)

    def _predict(self, x):
        out = self.model(x).squeeze(-1)
        return out.clamp(0, puzzle.MAX_COST).round()


class _V2Mixin:
    """Run-3 input encoding + residual reconstruction h = r_hat + md_sum."""

    def _encode(self, states):
        self._last_md_sums = puzzle.manhattan_sum(states).astype(np.float32)
        return np.concatenate([puzzle.one_hot(states), puzzle.per_cell_md(states)], axis=1)

    def __call__(self, states):
        residual = _TorchHeuristic.__call__(self, states)
        return residual + self._last_md_sums


class ClassifierV2Heuristic(_V2Mixin, ClassifierHeuristic):
    name = "classifier_v2"

    MAX_RESIDUAL = 44

    def _build_model(self, cfg):
        m = self._phase2_models()
        return m.PuzzleClassifierV2(width=cfg.get("width", 256), depth=cfg.get("depth", 4))

    def _predict(self, x):
        logits = self.model(x) / self.temperature
        cdf = self._torch.softmax(logits, dim=-1).cumsum(dim=-1)
        return (cdf < self.threshold).sum(dim=-1).clamp(0, self.MAX_RESIDUAL)


class RegressorV2Heuristic(_V2Mixin, RegressorHeuristic):
    name = "regressor_v2"

    MAX_RESIDUAL = 44

    def _build_model(self, cfg):
        m = self._phase2_models()
        return m.PuzzleRegressorV2(width=cfg.get("width", 256), depth=cfg.get("depth", 4))

    def _predict(self, x):
        out = self.model(x).squeeze(-1)
        return out.clamp(0, self.MAX_RESIDUAL).round()


HEURISTIC_REGISTRY = {
    "zero": ZeroHeuristic,
    "md": ManhattanHeuristic,
    "pdb": PDBHeuristic,
    "classifier": ClassifierHeuristic,
    "regressor": RegressorHeuristic,
    "classifier_v2": ClassifierV2Heuristic,
    "regressor_v2": RegressorV2Heuristic,
}


def build_heuristic(spec: dict) -> Heuristic:
    """{'type': 'classifier', 'checkpoint': ..., 'threshold': 0.1} -> Heuristic."""
    spec = dict(spec)
    kind = spec.pop("type")
    try:
        cls = HEURISTIC_REGISTRY[kind]
    except KeyError:
        raise ValueError(f"unknown heuristic type {kind!r}; known: {sorted(HEURISTIC_REGISTRY)}")
    return cls(**spec)
