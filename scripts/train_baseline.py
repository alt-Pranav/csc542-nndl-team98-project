#!/usr/bin/env python3
"""Train ResNet18-Encoder U-Net on Montgomery CXR (two-phase: frozen encoder, then full fine-tune)."""

import argparse
import copy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from loguru import logger

from unet_mech.config import DEFAULT_CFG
from unet_mech.data import build_dataloaders, download_montgomery
from unet_mech.interpret.hooks import print_architecture_reference, register_hooks, remove_hooks
from unet_mech.models import ResNet18UNet
from unet_mech.training import (
    DiceBCELoss,
    build_optimizer,
    plot_training_curves,
    save_checkpoint,
    train_one_epoch,
    validate,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Montgomery root (default: config)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Total training epochs (default: config num_epochs)",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Where to save best checkpoint",
    )
    p.add_argument(
        "--curves-out",
        type=str,
        default="training_curves.png",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="If set, cap epochs to 1+1 and skip heavy logging",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = copy.deepcopy(DEFAULT_CFG)
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.epochs is not None:
        cfg["num_epochs"] = args.epochs
    if args.ckpt:
        cfg["ckpt_path"] = args.ckpt
    if args.smoke:
        cfg["freeze_epochs"] = 1
        cfg["num_epochs"] = 2
        cfg["batch_size"] = min(cfg["batch_size"], 2)

    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    device = cfg["device"]
    logger.info(f"[main] Using device: {device}")

    root = download_montgomery(cfg["data_dir"])
    train_loader, val_loader, test_loader = build_dataloaders(
        root=str(root),
        batch_size=cfg["batch_size"],
        img_size=cfg["img_size"],
        train_frac=cfg["train_frac"],
        val_frac=cfg["val_frac"],
        seed=cfg["seed"],
    )

    model = ResNet18UNet(pretrained=True).to(device)
    print_architecture_reference(
        model, img_size=cfg["img_size"], device=device
    )
    register_hooks(model)

    criterion = DiceBCELoss(bce_weight=0.5)
    best_val_iou = 0.0
    best_ckpt = cfg["ckpt_path"]
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_dice": [],
    }

    # Training phase A: frozen encoder
    logger.info(
        f"\n[train] encoder frozen (initial) for {cfg['freeze_epochs']} epochs"
    )
    model.freeze_encoder()
    optimizer = build_optimizer(
        model, phase=1, encoder_lr=cfg["encoder_lr"], decoder_lr=cfg["decoder_lr"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    for epoch in range(1, cfg["freeze_epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_iou_, val_dice = validate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)
        for k, v in zip(
            ["train_loss", "val_loss", "val_iou", "val_dice"],
            [train_loss, val_loss, val_iou_, val_dice],
        ):
            history[k].append(v)
        logger.info(
            f"  [enc_frozen] Epoch [{epoch:>2}/{cfg['freeze_epochs']}] "
            f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
            f"val_IoU: {val_iou_:.4f}  val_Dice: {val_dice:.4f}"
        )
        if val_iou_ > best_val_iou:
            best_val_iou = val_iou_
            save_checkpoint(
                model, epoch, val_iou_, val_dice, best_ckpt, cfg
            )

    # Training phase B: unfreeze encoder
    remaining = cfg["num_epochs"] - cfg["freeze_epochs"]
    logger.info(
        f"\n[train] unfreeze encoder for {remaining} epochs "
        f"(enc_lr={cfg['encoder_lr']}, dec_lr={cfg['decoder_lr']})"
    )
    model.unfreeze_encoder()
    optimizer = build_optimizer(
        model, phase=2, encoder_lr=cfg["encoder_lr"], decoder_lr=cfg["decoder_lr"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    for epoch in range(cfg["freeze_epochs"] + 1, cfg["num_epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_iou_, val_dice = validate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)
        for k, v in zip(
            ["train_loss", "val_loss", "val_iou", "val_dice"],
            [train_loss, val_loss, val_iou_, val_dice],
        ):
            history[k].append(v)
        logger.info(
            f"  [finetune] Epoch [{epoch:>2}/{cfg['num_epochs']}] "
            f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
            f"val_IoU: {val_iou_:.4f}  val_Dice: {val_dice:.4f}"
        )
        if val_iou_ > best_val_iou:
            best_val_iou = val_iou_
            save_checkpoint(
                model, epoch, val_iou_, val_dice, best_ckpt, cfg
            )

    remove_hooks(model)
    plot_training_curves(
        history=history,
        freeze_epochs=cfg["freeze_epochs"],
        save_path=args.curves_out,
    )

    logger.info("\n[eval] Loading best checkpoint for test (task metrics: IoU / Dice) …")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    _, test_iou_, test_dice = validate(model, test_loader, criterion, device)
    logger.info(f"\n{'='*42}")
    logger.info(" Test set (task metrics: IoU, Dice)")
    logger.info(f"{'='*42}")
    logger.info(f"  IoU  (Jaccard): {test_iou_:.4f}")
    logger.info(f"  Dice (F1):      {test_dice:.4f}")
    logger.info(f"{'='*42}\n")
    logger.info(f"[main] Checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
