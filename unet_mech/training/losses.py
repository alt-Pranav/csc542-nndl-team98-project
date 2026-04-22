import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    """
    Combined Binary Cross-Entropy + Dice loss.
    BCE stabilises early training; Dice directly optimises overlap metrics
    and is robust to background / foreground imbalance.
    """

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = 1.0 - bce_weight
        self.bce = nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(
            pred, target
        ) + self.dice_weight * self._dice_loss(pred, target)

    @staticmethod
    def _dice_loss(
        pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        p = pred.view(-1)
        t = target.view(-1)
        return 1.0 - (2.0 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
