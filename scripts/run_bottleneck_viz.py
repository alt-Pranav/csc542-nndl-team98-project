#!/usr/bin/env python3
"""Bottleneck (enc_layer4) feature visualisation for a few samples."""

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
from unet_mech.interpret.bottleneck_viz import (
    save_bottleneck_channel_grid,
    save_bottleneck_with_overlay,
)
from unet_mech.models import ResNet18UNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Checkpoint .pth (state_dict under 'state_dict')",
    )
    p.add_argument("--out-dir", type=str, default="outputs/bottleneck_viz")
    p.add_argument("--max-ch", type=int, default=32)
    p.add_argument(
        "--selection", choices=["first", "top_abs_mean"], default="top_abs_mean"
    )
    p.add_argument("--overlay-ch", type=int, default=0, help="Also save single-ch overlay; set -1 to skip")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = copy.deepcopy(DEFAULT_CFG)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = download_montgomery(cfg["data_dir"])
    train_loader, val_loader, test_loader = build_dataloaders(
        root=str(root),
        batch_size=1,
        img_size=cfg["img_size"],
        train_frac=cfg["train_frac"],
        val_frac=cfg["val_frac"],
        seed=cfg["seed"],
    )
    # one sample from val
    it = iter(val_loader)
    image, _mask = next(it)
    model = ResNet18UNet(pretrained=True).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_bottleneck_channel_grid(
        model,
        image[0],
        device,
        save_path=out / "bottleneck_grid.png",
        max_channels=args.max_ch,
        selection=args.selection,  # type: ignore[arg-type]
    )
    if args.overlay_ch >= 0:
        save_bottleneck_with_overlay(
            model,
            image[0],
            device,
            save_path=out / f"overlay_ch{args.overlay_ch}.png",
            channel=args.overlay_ch,
        )
    logger.info(f"Wrote visualisations under {out}")


if __name__ == "__main__":
    main()
