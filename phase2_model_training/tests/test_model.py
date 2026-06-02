"""
Unit tests for Phase 2 model architecture and loss functions.
No real data required — uses random tensors only.

Run: pytest phase2_model_training/tests/test_model.py -v
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import PinballLoss, PuzzleClassifier, PuzzleRegressor

BATCH     = 8
INPUT_DIM = 256
NUM_COSTS = 81


def _rand_x() -> torch.Tensor:
    return torch.randn(BATCH, INPUT_DIM)


def _rand_costs() -> torch.Tensor:
    return torch.randint(0, NUM_COSTS, (BATCH,)).float()


# ── Parameter counts ───────────────────────────────────────────────────────────

def test_classifier_param_count():
    n = sum(p.numel() for p in PuzzleClassifier().parameters())
    assert n < 1_000_000, f"PuzzleClassifier has {n:,} params (limit 1M)"


def test_regressor_param_count():
    n = sum(p.numel() for p in PuzzleRegressor().parameters())
    assert n < 1_000_000, f"PuzzleRegressor has {n:,} params (limit 1M)"


# ── Forward output shapes ──────────────────────────────────────────────────────

def test_classifier_output_shape():
    logits = PuzzleClassifier()(_rand_x())
    assert logits.shape == (BATCH, NUM_COSTS), logits.shape


def test_regressor_output_shape():
    out = PuzzleRegressor()(_rand_x())
    assert out.shape == (BATCH, 1), out.shape


# ── Loss backward passes ───────────────────────────────────────────────────────

def test_classifier_loss_backward():
    model   = PuzzleClassifier()
    loss_fn = nn.CrossEntropyLoss()
    loss    = loss_fn(model(_rand_x()), _rand_costs().long())
    loss.backward()


def test_regressor_loss_backward():
    model   = PuzzleRegressor()
    loss_fn = PinballLoss(tau=0.3)
    loss    = loss_fn(model(_rand_x()), _rand_costs())
    loss.backward()


# ── PinballLoss asymmetry ──────────────────────────────────────────────────────

def test_pinball_asymmetry():
    """For τ < 0.5, overestimation must cost more than underestimation."""
    loss_fn = PinballLoss(tau=0.3)
    y       = torch.tensor([10.0])
    delta   = torch.tensor([2.0])

    over  = loss_fn(y + delta, y)   # pred=12, true=10 → overestimate by 2
    under = loss_fn(y - delta, y)   # pred=8,  true=10 → underestimate by 2

    assert over.item() > under.item(), (
        f"overestimate loss ({over.item():.4f}) should exceed "
        f"underestimate loss ({under.item():.4f}) for τ=0.3"
    )


# ── predict() API ──────────────────────────────────────────────────────────────

def test_classifier_predict_shape_and_range():
    preds = PuzzleClassifier().predict(_rand_x())
    assert preds.shape == (BATCH,)
    assert preds.min() >= 0 and preds.max() <= 80


def test_regressor_predict_shape_and_range():
    preds = PuzzleRegressor().predict(_rand_x())
    assert preds.shape == (BATCH,)
    assert preds.min() >= 0 and preds.max() <= 80
