#!/usr/bin/env python3
"""Find correlated bottleneck channels and validate clusters with ablation."""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from unet_mech.config import DEFAULT_CFG
from unet_mech.data import MontgomeryDataset, build_dataloaders, download_montgomery
from unet_mech.interpret.ablation import _mean_metrics_over_loader
from unet_mech.interpret.channel_correlation import (
    channel_correlation,
    channel_feature_matrix,
    cluster_ablation_rows,
    cluster_ids_for_focus,
    collect_layer_activations,
    nearest_neighbors,
    pca_coords,
    plot_cluster_overlays,
    plot_correlation_matrix,
    plot_pca_channels,
    read_ablation_csv,
    threshold_clusters,
    write_channel_summary,
    write_cluster_summary,
    write_rows,
)
from unet_mech.models import BabyUNet, ResNet18UNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", choices=["baby_unet", "resnet18_unet"], default="baby_unet")
    p.add_argument("--layer-name", default="bottleneck")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--corr-method", choices=["pearson", "cosine"], default="pearson")
    p.add_argument("--corr-threshold", type=float, default=0.85)
    p.add_argument("--use-abs-corr", action="store_true")
    p.add_argument("--top-k-neighbors", type=int, default=5)
    p.add_argument("--top-k-clusters", type=int, default=8)
    p.add_argument("--highlight-top-clusters", type=int, default=3)
    p.add_argument(
        "--highlight-clusters",
        default=None,
        help="Comma-separated cluster ids to highlight instead of the largest clusters",
    )
    p.add_argument(
        "--cluster-labels",
        default=None,
        help="Comma-separated id:label pairs, e.g. '0:top band,1:lung detector'",
    )
    p.add_argument("--focus-channels", default="16,118,105,60")
    p.add_argument("--ablation-csv", default=None)
    p.add_argument("--out-dir", default="outputs/channel_correlation")
    return p.parse_args()


def _resampling(name: str):
    return getattr(getattr(Image, "Resampling", Image), name)


class _BabyNoAugSubset(Dataset):
    def __init__(self, subset, img_size: int):
        self.subset = subset
        self.img_size = img_size
        self.image_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_tf = transforms.ToTensor()

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        ds = self.subset.dataset
        real_idx = self.subset.indices[idx]
        img_path, left_path, right_path = ds.samples[real_idx]
        size = (self.img_size, self.img_size)
        image = Image.open(img_path).convert("L").resize(size, _resampling("BILINEAR"))
        left = Image.open(left_path).convert("L").resize(size, _resampling("NEAREST"))
        right = Image.open(right_path).convert("L").resize(size, _resampling("NEAREST"))
        left_t = self.mask_tf(left)
        right_t = self.mask_tf(right)
        mask = ((left_t > 0.5) | (right_t > 0.5)).float()
        return self.image_tf(image), mask


def build_baby_loaders(root: Path, cfg: dict, batch_size: int) -> tuple[DataLoader, DataLoader]:
    full = MontgomeryDataset(
        root=str(root),
        img_size=cfg["img_size"],
        transform=transforms.ToTensor(),
        mask_transform=transforms.ToTensor(),
    )
    n = len(full)
    n_train = int(n * cfg["train_frac"])
    n_val = int(n * cfg["val_frac"])
    n_test = n - n_train - n_val
    gen = torch.Generator().manual_seed(cfg["seed"])
    _train, val, test = random_split(full, [n_train, n_val, n_test], generator=gen)
    val_ds = _BabyNoAugSubset(val, cfg["img_size"])
    test_ds = _BabyNoAugSubset(test, cfg["img_size"])
    return (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )


def load_model(args: argparse.Namespace, device: str) -> torch.nn.Module:
    if args.model == "baby_unet":
        model = BabyUNet().to(device)
    else:
        model = ResNet18UNet(pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def parse_channels(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_cluster_labels(value: str | None) -> dict[int, str]:
    if not value:
        return {}
    labels: dict[int, str] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        cluster_id, label = item.split(":", maxsplit=1)
        labels[int(cluster_id.strip())] = label.strip()
    return labels


def write_target_report(
    path: Path,
    corr,
    clusters,
    channels: list[int],
) -> None:
    from unet_mech.interpret.channel_correlation import cluster_lookup

    lookup = cluster_lookup(clusters)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["channel_a", "channel_b", "corr", "same_cluster", "cluster_a", "cluster_b"],
        )
        writer.writeheader()
        for i, channel_a in enumerate(channels):
            for channel_b in channels[i + 1 :]:
                if channel_a >= corr.shape[0] or channel_b >= corr.shape[0]:
                    continue
                cluster_a = lookup[channel_a]
                cluster_b = lookup[channel_b]
                writer.writerow(
                    {
                        "channel_a": channel_a,
                        "channel_b": channel_b,
                        "corr": float(corr[channel_a, channel_b]),
                        "same_cluster": cluster_a == cluster_b,
                        "cluster_a": cluster_a,
                        "cluster_b": cluster_b,
                    }
                )


def main() -> None:
    args = parse_args()
    cfg = copy.deepcopy(DEFAULT_CFG)
    batch_size = args.batch_size or cfg["batch_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = download_montgomery(cfg["data_dir"])
    if args.model == "baby_unet":
        val_loader, test_loader = build_baby_loaders(root, cfg, batch_size)
        image_mode = "gray"
    else:
        _train_loader, val_loader, test_loader = build_dataloaders(
            root=str(root),
            batch_size=batch_size,
            img_size=cfg["img_size"],
            train_frac=cfg["train_frac"],
            val_frac=cfg["val_frac"],
            seed=cfg["seed"],
        )
        image_mode = "imagenet"
    loader = val_loader if args.split == "val" else test_loader

    model = load_model(args, device)
    logger.info(f"Collecting activations from {args.model}:{args.layer_name} on {args.split}")
    activations = collect_layer_activations(
        model,
        loader,
        device,
        layer_name=args.layer_name,
        max_batches=args.max_batches,
    )
    features = channel_feature_matrix(activations)
    corr = channel_correlation(features, method=args.corr_method)
    clusters = threshold_clusters(
        corr,
        threshold=args.corr_threshold,
        use_abs=args.use_abs_corr,
    )
    coords = pca_coords(corr)
    neighbors = nearest_neighbors(corr, top_k=args.top_k_neighbors)
    ablation = read_ablation_csv(args.ablation_csv)
    focus_channels = parse_channels(args.focus_channels)
    highlight_cluster_ids = (
        parse_channels(args.highlight_clusters)
        if args.highlight_clusters is not None
        else None
    )
    cluster_labels = parse_cluster_labels(args.cluster_labels)

    np.save(out_dir / "channel_correlation.npy", corr)
    write_channel_summary(out_dir / "channel_correlation.csv", corr, clusters, neighbors, ablation)
    write_cluster_summary(out_dir / "channel_clusters.csv", clusters, corr, ablation)
    write_target_report(out_dir / "focus_channel_pairs.csv", corr, clusters, focus_channels)
    plot_correlation_matrix(corr, clusters, out_dir / "correlation_matrix.png")
    plot_pca_channels(
        coords,
        clusters,
        out_dir / "pca_channels.png",
        highlight_channels=focus_channels,
        ablation=ablation,
        top_n_clusters=args.highlight_top_clusters,
        highlight_cluster_ids=highlight_cluster_ids,
        cluster_labels=cluster_labels,
    )

    sample_images, _sample_masks = next(iter(loader))
    plot_cluster_overlays(
        activations,
        sample_images[0],
        clusters,
        out_dir / "cluster_overlays.png",
        max_clusters=min(12, len(clusters)),
        image_mode=image_mode,
    )

    baseline = _mean_metrics_over_loader(model, loader, device)
    cluster_ids = cluster_ids_for_focus(
        clusters,
        ablation,
        focus_channels,
        top_k=args.top_k_clusters,
    )
    if highlight_cluster_ids is None:
        highlighted = {cluster.cluster_id for cluster in clusters[: args.highlight_top_clusters]}
    else:
        highlighted = set(highlight_cluster_ids)
    cluster_ids = sorted(set(cluster_ids) | highlighted)
    rows = cluster_ablation_rows(
        model,
        loader,
        device,
        args.layer_name,
        baseline.mean_iou,
        baseline.mean_dice,
        clusters,
        cluster_ids,
    )
    write_rows(out_dir / "cluster_ablation.csv", rows)

    logger.info(
        f"Done. channels={corr.shape[0]} clusters={len(clusters)} "
        f"largest_cluster={max(c.size for c in clusters)} out={out_dir}"
    )
    for channel in focus_channels:
        if 0 <= channel < corr.shape[0]:
            nn = ", ".join(f"{idx}:{score:.3f}" for idx, score in neighbors[channel])
            logger.info(f"focus ch {channel}: nearest {nn}")


if __name__ == "__main__":
    main()
