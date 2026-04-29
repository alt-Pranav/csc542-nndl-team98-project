#!/usr/bin/env python3
"""Compute activation sparsity scores for each bottleneck channel."""

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
from unet_mech.interpret.sparsity import compute_sparsity_scores
from unet_mech.models import BabyUNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out-csv", type=str, default="outputs/sparsity_scores.csv")
    p.add_argument("--layer-name", type=str, default="enc_layer4")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = copy.deepcopy(DEFAULT_CFG)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = download_montgomery(cfg["data_dir"])
    _, val_loader, _ = build_dataloaders(
        root=str(root),
        batch_size=cfg["batch_size"],
        img_size=cfg["img_size"],
        train_frac=cfg["train_frac"],
        val_frac=cfg["val_frac"],
        seed=cfg["seed"],
        num_workers=0,
    )

    model = BabyUNet(pretrained=False).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["state_dict"])

    logger.info(f"Computing sparsity scores on val split, layer={args.layer_name}")
    rows = compute_sparsity_scores(
        model, val_loader, device,
        layer_name=args.layer_name,
        csv_path=args.out_csv,
    )
    logger.info(f"Done. {len(rows)} channels scored -> {args.out_csv}")


if __name__ == "__main__":
    main()
