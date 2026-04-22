import torch
import torch.nn as nn
from loguru import logger

from unet_mech.models.resnet18_unet import ResNet18UNet


def save_checkpoint(
    model: ResNet18UNet,
    epoch: int,
    val_iou: float,
    val_dice: float,
    path: str,
    cfg: dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "val_iou": val_iou,
            "val_dice": val_dice,
            "cfg": cfg,
        },
        path,
    )
    logger.info(f"    ✓ Checkpoint saved → {path}  (IoU={val_iou:.4f})")
