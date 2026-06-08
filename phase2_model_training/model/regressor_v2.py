import torch
import torch.nn as nn

# Maximum expected residual r(s) = h*(s) − MD_sum_excl_blank(s).
# Empirical max over 100M records: 34.  Clamping at 44 gives 10 moves of
# headroom while matching the classifier's NUM_RESIDUALS − 1 so both models
# share the same effective prediction range for fair comparison.
MAX_RESIDUAL = 44


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


class PuzzleRegressorV2(nn.Module):
    """
    Scalar residual regressor trained with PinballLoss.

    Input:  272-dim vector (256 one-hot + 16 per-cell Manhattan distances).
    Output: predicted residual r_hat(s) = h_hat(s) − MD_sum_excl_blank(s).

    Phase 3 reconstructs h_hat(s) = predict(x) + manhattan_sum(state).
    PinballLoss τ=0.3 penalizes overestimates 2.3× more, biasing the model
    toward the 30th percentile of the residual distribution.
    """

    def __init__(self, width: int = 256, depth: int = 4, input_dim: int = 272) -> None:
        super().__init__()
        self.proj   = nn.Sequential(nn.Linear(input_dim, width), nn.ReLU())
        self.blocks = nn.Sequential(*[_ResBlock(width) for _ in range(depth)])
        self.head   = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw residual prediction [B, 1]."""
        return self.head(self.blocks(self.proj(x)))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted residual as int tensor [B], clamped to [0, MAX_RESIDUAL]."""
        return self.forward(x).squeeze(-1).clamp(0, MAX_RESIDUAL).round().long()
