import torch
import torch.nn as nn

MAX_COST = 80


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


class PuzzleRegressor(nn.Module):
    """
    Scalar cost regressor trained with pinball (quantile) loss.

    Architecture: Linear projection → depth residual blocks → scalar head.
    ~594K parameters at default width=256, depth=4 — well under 1M.

    forward()  returns raw scalar predictions [B, 1].
    predict()  returns round(clamp(output, 0, 80)) as int tensor [B].
    """

    def __init__(self, width: int = 256, depth: int = 4) -> None:
        super().__init__()
        self.proj   = nn.Sequential(nn.Linear(256, width), nn.ReLU())
        self.blocks = nn.Sequential(*[_ResBlock(width) for _ in range(depth)])
        self.head   = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw scalar prediction [B, 1]."""
        return self.head(self.blocks(self.proj(x)))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted cost as int tensor [B], clamped to [0, 80]."""
        return self.forward(x).squeeze(-1).clamp(0, MAX_COST).round().long()
