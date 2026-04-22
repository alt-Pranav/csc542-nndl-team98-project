from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from unet_mech.metrics.segmentation import dice, iou
from unet_mech.models.resnet18_unet import ResNet18UNet


@dataclass
class SegmentationEval:
    mean_iou: float
    mean_dice: float
    n_batches: int


@torch.no_grad()
def _mean_metrics_over_loader(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> SegmentationEval:
    model.eval()
    ious, dices = [], []
    n_batches = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        preds = model(images)
        pred_bin = (preds > 0.5).float()
        ious.append(iou(pred_bin, masks).item())
        dices.append(dice(pred_bin, masks).item())
        n_batches += 1
    return SegmentationEval(
        mean_iou=float(np.mean(ious)),
        mean_dice=float(np.mean(dices)),
        n_batches=n_batches,
    )


def _bottleneck_ablation_hook(channels: set[int]):
    def hook(module, inp, out: torch.Tensor):
        m = out.clone()
        for c in channels:
            if 0 <= c < m.shape[1]:
                m[:, c] = 0.0
        return m

    return hook


@torch.no_grad()
def evaluate_bottleneck_channel_ablation(
    model: ResNet18UNet,
    loader: DataLoader,
    device: str,
    channel_indices: Iterable[int],
) -> SegmentationEval:
    """
    Zeros the given `encoder_layer4` (bottleneck) channels during the forward
    pass and returns mean batch IoU / Dice on the loader.
    """
    chans = {int(c) for c in channel_indices}
    handle = model.encoder_layer4.register_forward_hook(
        _bottleneck_ablation_hook(chans)
    )
    try:
        return _mean_metrics_over_loader(model, loader, device)
    finally:
        handle.remove()


@dataclass
class AblationSweepResult:
    baseline: SegmentationEval
    rows: list[dict]


@torch.no_grad()
def ablation_sweep_bottleneck(
    model: ResNet18UNet,
    loader: DataLoader,
    device: str,
    channel_ids: List[int] | range,
    csv_path: str | Path | None = None,
) -> AblationSweepResult:
    """
    One baseline pass, then one forward pass per channel with only that channel
    ablated. Writes optional CSV: channel, iou, dice, delta_iou, delta_dice.
    """
    baseline = _mean_metrics_over_loader(model, loader, device)
    logger.info(
        f"[ablation] baseline  IoU={baseline.mean_iou:.4f}  Dice={baseline.mean_dice:.4f} "
        f"({baseline.n_batches} batches)"
    )

    rows: list[dict] = []
    for c in channel_ids:
        ev = evaluate_bottleneck_channel_ablation(model, loader, device, [c])
        row = {
            "channel": c,
            "iou": ev.mean_iou,
            "dice": ev.mean_dice,
            "delta_iou": ev.mean_iou - baseline.mean_iou,
            "delta_dice": ev.mean_dice - baseline.mean_dice,
        }
        rows.append(row)
        logger.info(
            f"  ch {c:>4}  IoU={ev.mean_iou:.4f}  ΔIoU={row['delta_iou']:+.4f}  "
            f"Dice={ev.mean_dice:.4f}  ΔDice={row['delta_dice']:+.4f}"
        )

    if csv_path is not None:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["channel", "iou", "dice", "delta_iou", "delta_dice"]
            )
            w.writeheader()
            w.writerows(rows)
        logger.info(f"[ablation] Wrote {csv_path}")

    return AblationSweepResult(baseline=baseline, rows=rows)
