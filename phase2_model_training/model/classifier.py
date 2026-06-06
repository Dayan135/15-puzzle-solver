import torch
import torch.nn as nn

NUM_COSTS = 81  # discrete classes 0..80


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


class PuzzleClassifier(nn.Module):
    """
    81-class classifier predicting P(cost = k) for k ∈ {0..80}.

    Architecture: Linear projection → depth residual blocks → 81-way head.
    ~615K parameters at default width=256, depth=4 — well under 1M.

    forward()  returns logits [B, 81].
    predict()  returns argmax cost as int tensor [B].
    """

    def __init__(self, width: int = 256, depth: int = 4) -> None:
        super().__init__()
        self.proj   = nn.Sequential(nn.Linear(256, width), nn.ReLU())
        self.blocks = nn.Sequential(*[_ResBlock(width) for _ in range(depth)])
        self.head   = nn.Linear(width, NUM_COSTS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits [B, 81]."""
        return self.head(self.blocks(self.proj(x)))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted cost as int tensor [B] (argmax over class logits)."""
        return self.forward(x).argmax(dim=-1)

    @torch.no_grad()
    def predict_quantile(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Returns the smallest k such that P(cost ≤ k) ≥ threshold.

        Unlike predict(), this uses the full CDF rather than the mode, which
        eliminates catastrophic tail overestimates and gives a tunable
        admissibility dial without retraining.  threshold=0.5 → median.
        """
        cdf = torch.softmax(self.forward(x), dim=-1).cumsum(dim=-1)   # [B, 81]
        return (cdf < threshold).sum(dim=-1).clamp(max=80).long()     # [B]
