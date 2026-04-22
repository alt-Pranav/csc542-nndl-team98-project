from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from unet_mech.metrics.segmentation import dice, iou
from unet_mech.models.resnet18_unet import ResNet18UNet


def build_optimizer(
    model: ResNet18UNet,
    phase: int,
    encoder_lr: float,
    decoder_lr: float,
) -> torch.optim.Optimizer:
    """
    phase 1: decoder only (encoder should be frozen).
    phase 2: two param groups for encoder and decoder.
    """
    if phase == 1:
        return torch.optim.Adam(model.decoder_parameters(), lr=decoder_lr)
    return torch.optim.Adam(
        [
            {"params": model.encoder_parameters(), "lr": encoder_lr},
            {"params": model.decoder_parameters(), "lr": decoder_lr},
        ]
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    iou_scores, dice_scores = [], []

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        preds = model(images)
        total_loss += criterion(preds, masks).item() * images.size(0)

        pred_bin = (preds > 0.5).float()
        iou_scores.append(iou(pred_bin, masks).item())
        dice_scores.append(dice(pred_bin, masks).item())

    return (
        total_loss / len(loader.dataset),
        float(np.mean(iou_scores)),
        float(np.mean(dice_scores)),
    )
