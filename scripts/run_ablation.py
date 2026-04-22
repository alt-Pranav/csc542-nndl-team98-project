#!/usr/bin/env python3
"""Bottleneck channel ablation: measure IoU/Dice when individual channels are zeroed."""

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
from unet_mech.interpret.ablation import ablation_sweep_bottleneck
from unet_mech.models import ResNet18UNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument(
        "--channels",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated single-channel ablations, or 'range(0,16)'-style: use start-end as two args",
    )
    p.add_argument("--from-ch", type=int, default=None)
    p.add_argument("--to-ch", type=int, default=None, help="exclusive end for a range")
    p.add_argument("--out-csv", type=str, default="outputs/ablation_sweep.csv")
    p.add_argument("--split", choices=["val", "test"], default="val")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = copy.deepcopy(DEFAULT_CFG)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.from_ch is not None and args.to_ch is not None:
        ch_list = list(range(args.from_ch, args.to_ch))
    else:
        ch_list = [int(x.strip()) for x in args.channels.split(",") if x.strip()]

    root = download_montgomery(cfg["data_dir"])
    train_loader, val_loader, test_loader = build_dataloaders(
        root=str(root),
        batch_size=cfg["batch_size"],
        img_size=cfg["img_size"],
        train_frac=cfg["train_frac"],
        val_frac=cfg["val_frac"],
        seed=cfg["seed"],
    )
    loader = val_loader if args.split == "val" else test_loader

    model = ResNet18UNet(pretrained=True).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["state_dict"])

    logger.info(
        f"Running ablation on {args.split} split, channels: {ch_list[:20]}{'...' if len(ch_list) > 20 else ''}"
    )
    res = ablation_sweep_bottleneck(
        model, loader, device, ch_list, csv_path=args.out_csv
    )
    logger.info(
        f"Done. baseline IoU={res.baseline.mean_iou:.4f}  rows={len(res.rows)}  csv={args.out_csv}"
    )


if __name__ == "__main__":
    main()
