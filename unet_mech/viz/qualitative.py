from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from unet_mech.config import IMAGENET_MEAN, IMAGENET_STD


def denormalize_imagenet(t: torch.Tensor) -> np.ndarray:
    """[3,H,W] normalised → [H,W,3] uint8 (ImageNet)."""
    mean = torch.tensor(IMAGENET_MEAN, device=t.device, dtype=t.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=t.device, dtype=t.dtype).view(3, 1, 1)
    img = t * std + mean
    img = img.clamp(0.0, 1.0)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def plot_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    n: int = 5,
    save_path: str = "predictions.png",
) -> None:
    """
    n × 3 grid: ground-truth mask | denormalised image | prediction (threshold 0.5).
    """
    model.eval()
    images_out, gt_masks, pred_masks = [], [], []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            preds = model(images)

            for i in range(images.size(0)):
                if len(images_out) >= n:
                    break
                images_out.append(denormalize_imagenet(images[i]))
                gt_masks.append(masks[i, 0].cpu().numpy())
                pred_masks.append(
                    (preds[i, 0].cpu().numpy() > 0.5).astype(np.uint8)
                )

            if len(images_out) >= n:
                break

    actual_n = len(images_out)
    fig, axes = plt.subplots(
        actual_n, 3, figsize=(9, actual_n * 3), constrained_layout=True
    )
    if actual_n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Ground Truth Mask", "Original Image", "Predicted Mask"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=6)

    for row in range(actual_n):
        axes[row, 0].imshow(gt_masks[row], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(images_out[row])
        axes[row, 2].imshow(pred_masks[row], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"Sample {row + 1}", fontsize=9, labelpad=4)
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.suptitle("ResNet18-UNet — Prediction Samples", fontsize=13, fontweight="bold")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[plot] Prediction grid saved → {save_path}")
