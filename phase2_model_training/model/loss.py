import torch
import torch.nn as nn


class PinballLoss(nn.Module):
    """
    Quantile (pinball) loss for admissibility-biased regression.

    L(y, ŷ) = τ · max(y − ŷ, 0) + (1 − τ) · max(ŷ − y, 0)

    τ < 0.5 penalizes overestimation more than underestimation, biasing the
    model toward admissible (≤ true cost) predictions.

    Default τ=0.3 → overestimation penalized 2.3× heavier than underestimation.
    The model learns the τ-th quantile of the cost distribution.
    """

    def __init__(self, tau: float = 0.3) -> None:
        super().__init__()
        if not (0.0 < tau < 1.0):
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        self.tau = tau

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred     = pred.squeeze(-1)
        residual = target - pred
        loss     = torch.where(residual >= 0, self.tau * residual, (self.tau - 1.0) * residual)
        return loss.mean()
