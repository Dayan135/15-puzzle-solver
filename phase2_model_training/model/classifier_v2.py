import torch
import torch.nn as nn

# Residual r(s) = h*(s) − MD_sum_excl_blank(s).
# Empirical max over 100M records: 34.  45 classes cover [0, 44], giving 10
# bins of headroom above the observed max.  Halved vs the 81-class h* classifier
# → sharper CDF calibration over a smaller output range.
NUM_RESIDUALS = 45


class _ResBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1  = nn.Linear(width, width)
        self.fc2  = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.act  = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        return self.norm(x + self.fc2(h))


class PuzzleClassifierV2(nn.Module):
    """
    45-class classifier predicting P(residual = k) for k ∈ {0..44}.

    Input: 272-dim vector (256 one-hot + 16 per-cell Manhattan distances).
    Output: logits [B, 45] over residual classes.

    predict_quantile returns the smallest k s.t. P(residual ≤ k) ≥ threshold,
    with optional temperature scaling to smooth overconfident distributions.
    Phase 3 reconstructs h_hat(s) = predict_quantile(x) + manhattan_sum(state).
    """

    def __init__(self, width: int = 256, depth: int = 4, input_dim: int = 272) -> None:
        super().__init__()
        self.num_residuals = NUM_RESIDUALS
        self.proj   = nn.Sequential(nn.Linear(input_dim, width), nn.ReLU())
        self.blocks = nn.Sequential(*[_ResBlock(width) for _ in range(depth)])
        self.head   = nn.Linear(width, NUM_RESIDUALS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns residual logits [B, 45]."""
        return self.head(self.blocks(self.proj(x)))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted residual as int tensor [B] (argmax)."""
        return self.forward(x).argmax(dim=-1)

    @torch.no_grad()
    def predict_quantile(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Returns smallest k s.t. P(residual ≤ k) ≥ threshold.

        temperature > 1 spreads the distribution, reducing overconfident
        single-bin peaks and lowering overestimate_max without retraining.
        """
        logits = self.forward(x)
        if temperature != 1.0:
            logits = logits / temperature
        cdf = torch.softmax(logits, dim=-1).cumsum(dim=-1)          # [B, 45]
        return (cdf < threshold).sum(dim=-1).clamp(max=NUM_RESIDUALS - 1).long()
