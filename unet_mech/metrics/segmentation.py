import torch


def iou(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Intersection-over-Union (Jaccard) averaged over a batch."""
    p = pred.view(pred.size(0), -1).float()
    t = target.view(target.size(0), -1).float()
    inter = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1) - inter
    return ((inter + eps) / (union + eps)).mean()


def dice(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Sørensen–Dice coefficient averaged over a batch."""
    p = pred.view(pred.size(0), -1).float()
    t = target.view(target.size(0), -1).float()
    inter = (p * t).sum(dim=1)
    return ((2.0 * inter + eps) / (p.sum(dim=1) + t.sum(dim=1) + eps)).mean()
