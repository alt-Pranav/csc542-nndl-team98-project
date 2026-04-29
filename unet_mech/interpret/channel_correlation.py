from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from unet_mech.interpret.ablation import evaluate_bottleneck_channel_ablation
from unet_mech.viz.qualitative import denormalize_imagenet


@dataclass
class ChannelCluster:
    cluster_id: int
    channels: list[int]

    @property
    def size(self) -> int:
        return len(self.channels)


def named_hook_module(model: nn.Module, layer_name: str) -> nn.Module:
    if hasattr(model, "hook_target_layers"):
        for name, module in model.hook_target_layers():
            if name == layer_name:
                return module
    if hasattr(model, layer_name):
        module = getattr(model, layer_name)
        if isinstance(module, nn.Module):
            return module
    raise KeyError(f"Could not find layer {layer_name!r} on {type(model).__name__}")


@torch.no_grad()
def collect_layer_activations(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    layer_name: str,
    max_batches: int | None = None,
) -> torch.Tensor:
    """Return activations as a CPU tensor shaped [N, C, H, W]."""
    model.eval()
    module = named_hook_module(model, layer_name)
    batches: list[torch.Tensor] = []

    def hook(_module, _inp, out: torch.Tensor) -> None:
        batches.append(out.detach().cpu())

    handle = module.register_forward_hook(hook)
    try:
        for batch_idx, (images, _masks) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            _ = model(images)
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break
    finally:
        handle.remove()

    if not batches:
        raise ValueError("No activations collected; loader may be empty")
    return torch.cat(batches, dim=0)


def channel_feature_matrix(activations: torch.Tensor) -> np.ndarray:
    """Flatten [N, C, H, W] activations into [C, N*H*W] channel profiles."""
    if activations.dim() != 4:
        raise ValueError(f"Expected [N,C,H,W] activations, got {tuple(activations.shape)}")
    x = activations.float().permute(1, 0, 2, 3).reshape(activations.shape[1], -1)
    return x.numpy()


def channel_correlation(features: np.ndarray, method: str = "pearson") -> np.ndarray:
    if method not in {"pearson", "cosine"}:
        raise ValueError("method must be 'pearson' or 'cosine'")
    x = features.astype(np.float64, copy=True)
    if method == "pearson":
        x -= x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-12)
    corr = x @ x.T
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def pca_coords(matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project channels to 2D by SVD over centered channel descriptors."""
    x = matrix.astype(np.float64, copy=True)
    x -= x.mean(axis=0, keepdims=True)
    _u, _s, vh = np.linalg.svd(x, full_matrices=False)
    coords = x @ vh[:n_components].T
    if coords.shape[1] < n_components:
        coords = np.pad(coords, ((0, 0), (0, n_components - coords.shape[1])))
    return coords[:, :n_components]


def threshold_clusters(corr: np.ndarray, threshold: float = 0.85, use_abs: bool = False) -> list[ChannelCluster]:
    sim = np.abs(corr) if use_abs else corr
    n = sim.shape[0]
    seen = np.zeros(n, dtype=bool)
    clusters: list[ChannelCluster] = []

    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        members: list[int] = []
        while stack:
            cur = stack.pop()
            members.append(cur)
            neighbors = np.flatnonzero((sim[cur] >= threshold) & (~seen))
            for neighbor in neighbors.tolist():
                seen[neighbor] = True
                stack.append(neighbor)
        clusters.append(ChannelCluster(cluster_id=len(clusters), channels=sorted(members)))

    clusters.sort(key=lambda c: (-c.size, c.channels[0]))
    return [
        ChannelCluster(cluster_id=i, channels=cluster.channels)
        for i, cluster in enumerate(clusters)
    ]


def cluster_lookup(clusters: Iterable[ChannelCluster]) -> dict[int, int]:
    lookup: dict[int, int] = {}
    for cluster in clusters:
        for channel in cluster.channels:
            lookup[channel] = cluster.cluster_id
    return lookup


def nearest_neighbors(corr: np.ndarray, top_k: int = 5) -> dict[int, list[tuple[int, float]]]:
    neighbors: dict[int, list[tuple[int, float]]] = {}
    for channel in range(corr.shape[0]):
        order = np.argsort(-corr[channel])
        picked = [idx for idx in order.tolist() if idx != channel][:top_k]
        neighbors[channel] = [(idx, float(corr[channel, idx])) for idx in picked]
    return neighbors


def read_ablation_csv(path: str | Path | None) -> dict[int, dict[str, float]]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    rows: dict[int, dict[str, float]] = {}
    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            channel = int(row["channel"])
            rows[channel] = {
                key: float(value)
                for key, value in row.items()
                if key != "channel" and value != ""
            }
    return rows


def write_channel_summary(
    path: str | Path,
    corr: np.ndarray,
    clusters: list[ChannelCluster],
    neighbors: dict[int, list[tuple[int, float]]],
    ablation: dict[int, dict[str, float]] | None = None,
) -> None:
    ablation = ablation or {}
    lookup = cluster_lookup(clusters)
    sizes = {cluster.cluster_id: cluster.size for cluster in clusters}
    fieldnames = [
        "channel",
        "cluster_id",
        "cluster_size",
        "nearest_neighbors",
        "nearest_neighbor_corr",
        "mean_cluster_corr",
        "delta_iou",
        "delta_dice",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for channel in range(corr.shape[0]):
            cluster_id = lookup[channel]
            members = [c for c in clusters[cluster_id].channels if c != channel]
            mean_cluster_corr = (
                float(np.mean([corr[channel, c] for c in members]))
                if members
                else 1.0
            )
            nn = neighbors[channel]
            writer.writerow(
                {
                    "channel": channel,
                    "cluster_id": cluster_id,
                    "cluster_size": sizes[cluster_id],
                    "nearest_neighbors": " ".join(str(idx) for idx, _score in nn),
                    "nearest_neighbor_corr": " ".join(f"{score:.6f}" for _idx, score in nn),
                    "mean_cluster_corr": mean_cluster_corr,
                    "delta_iou": ablation.get(channel, {}).get("delta_iou", ""),
                    "delta_dice": ablation.get(channel, {}).get("delta_dice", ""),
                }
            )


def write_cluster_summary(
    path: str | Path,
    clusters: list[ChannelCluster],
    corr: np.ndarray,
    ablation: dict[int, dict[str, float]] | None = None,
) -> None:
    ablation = ablation or {}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cluster_id",
                "size",
                "channels",
                "mean_internal_corr",
                "mean_delta_iou",
                "min_delta_iou",
                "max_delta_iou",
            ],
        )
        writer.writeheader()
        for cluster in clusters:
            channels = cluster.channels
            if len(channels) > 1:
                sub = corr[np.ix_(channels, channels)]
                tri = sub[np.triu_indices_from(sub, k=1)]
                mean_corr = float(np.mean(tri))
            else:
                mean_corr = 1.0
            deltas = [
                ablation[channel]["delta_iou"]
                for channel in channels
                if channel in ablation and "delta_iou" in ablation[channel]
            ]
            writer.writerow(
                {
                    "cluster_id": cluster.cluster_id,
                    "size": cluster.size,
                    "channels": " ".join(str(c) for c in channels),
                    "mean_internal_corr": mean_corr,
                    "mean_delta_iou": float(np.mean(deltas)) if deltas else "",
                    "min_delta_iou": float(np.min(deltas)) if deltas else "",
                    "max_delta_iou": float(np.max(deltas)) if deltas else "",
                }
            )


def plot_correlation_matrix(
    corr: np.ndarray,
    clusters: list[ChannelCluster],
    path: str | Path,
) -> None:
    order = [channel for cluster in clusters for channel in cluster.channels]
    ordered = corr[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(ordered, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="channel correlation")
    ax.set_title("Bottleneck Channel Correlation (cluster ordered)")
    ax.set_xlabel("channel order by cluster")
    ax.set_ylabel("channel order by cluster")
    offset = 0
    for cluster in clusters:
        offset += cluster.size
        ax.axhline(offset - 0.5, color="black", linewidth=0.4)
        ax.axvline(offset - 0.5, color="black", linewidth=0.4)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_channels(
    coords: np.ndarray,
    clusters: list[ChannelCluster],
    path: str | Path,
    highlight_channels: Iterable[int] = (),
    ablation: dict[int, dict[str, float]] | None = None,
    top_n_clusters: int = 3,
    highlight_cluster_ids: Iterable[int] | None = None,
    cluster_labels: dict[int, str] | None = None,
) -> None:
    ablation = ablation or {}
    cluster_labels = cluster_labels or {}
    lookup = cluster_lookup(clusters)
    by_id = {cluster.cluster_id: cluster for cluster in clusters}
    sizes = [
        45 + 2400 * abs(ablation.get(channel, {}).get("delta_iou", 0.0))
        for channel in range(coords.shape[0])
    ]
    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=sizes,
        c="#c7c7c7",
        alpha=0.45,
        label="other clusters",
        linewidths=0,
    )

    if highlight_cluster_ids is None:
        highlighted_clusters = clusters[:top_n_clusters]
    else:
        highlighted_clusters = [
            by_id[cluster_id]
            for cluster_id in highlight_cluster_ids
            if cluster_id in by_id
        ]
    palette = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    highlighted_lookup = {
        cluster.cluster_id: rank
        for rank, cluster in enumerate(highlighted_clusters)
    }
    for rank, cluster in enumerate(highlighted_clusters, start=1):
        channels = cluster.channels
        color = palette[(rank - 1) % len(palette)]
        label = cluster_labels.get(cluster.cluster_id, f"cluster {cluster.cluster_id}")
        xy = coords[channels]
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=[sizes[channel] + 20 for channel in channels],
            c=color,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.35,
            label=f"{label}: cluster {cluster.cluster_id} (n={cluster.size})",
        )

        center = xy.mean(axis=0)
        span = np.maximum(xy.max(axis=0) - xy.min(axis=0), 0.35)
        ellipse = Ellipse(
            xy=center,
            width=span[0] + 0.55,
            height=span[1] + 0.55,
            angle=0,
            fill=False,
            color=color,
            linewidth=2.0,
            linestyle="--",
        )
        ax.add_patch(ellipse)
        ax.annotate(
            f"{label}\ncluster {cluster.cluster_id}, n={cluster.size}",
            center,
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=color,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "alpha": 0.85},
        )

    for channel in highlight_channels:
        if 0 <= channel < coords.shape[0]:
            cluster_id = lookup[channel]
            rank = highlighted_lookup.get(cluster_id)
            color = palette[(rank - 1) % len(palette)] if rank is not None else "black"
            ax.annotate(
                str(channel),
                (coords[channel, 0], coords[channel, 1]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color=color,
            )
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.set_title("Channel PCA from Activation-Correlation Profiles (Semantic Clusters Highlighted)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_overlays(
    activations: torch.Tensor,
    image: torch.Tensor,
    clusters: list[ChannelCluster],
    path: str | Path,
    max_clusters: int = 12,
    image_mode: str = "imagenet",
) -> None:
    if activations.dim() != 4:
        raise ValueError("activations must be [N,C,H,W]")
    acts = activations[0]
    h, w = image.shape[-2:]
    clusters_to_plot = clusters[:max_clusters]
    fig, axes = plt.subplots(
        len(clusters_to_plot),
        3,
        figsize=(9, 3 * len(clusters_to_plot)),
        constrained_layout=True,
    )
    if len(clusters_to_plot) == 1:
        axes = np.expand_dims(axes, axis=0)

    if image_mode == "imagenet":
        base = denormalize_imagenet(image).astype(np.float32) / 255.0
    else:
        gray = (image[0].detach().cpu().numpy() * 0.5 + 0.5).clip(0.0, 1.0)
        base = np.repeat(gray[..., None], 3, axis=2)

    for row, cluster in enumerate(clusters_to_plot):
        cluster_act = acts[cluster.channels].mean(dim=0, keepdim=True).unsqueeze(0)
        up = F.interpolate(
            cluster_act.float(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        up = (up - up.min()) / (up.max() - up.min() + 1e-8)
        axes[row, 0].imshow(base)
        axes[row, 0].set_title(f"cluster {cluster.cluster_id}: {cluster.channels[:8]}")
        axes[row, 1].imshow(up, cmap="magma", vmin=0, vmax=1)
        axes[row, 1].set_title("mean activation")
        axes[row, 2].imshow(base)
        axes[row, 2].imshow(up, cmap="magma", alpha=0.45, vmin=0, vmax=1)
        axes[row, 2].set_title("overlay")
        for col in range(3):
            axes[row, col].axis("off")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def cluster_ablation_rows(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    layer_name: str,
    baseline_iou: float,
    baseline_dice: float,
    clusters: list[ChannelCluster],
    cluster_ids: Iterable[int],
) -> list[dict]:
    by_id = {cluster.cluster_id: cluster for cluster in clusters}
    rows: list[dict] = []
    for cluster_id in cluster_ids:
        cluster = by_id[cluster_id]
        ev = evaluate_bottleneck_channel_ablation(
            model,
            loader,
            device,
            cluster.channels,
            layer_name=layer_name,
        )
        rows.append(
            {
                "cluster_id": cluster.cluster_id,
                "size": cluster.size,
                "channels": " ".join(str(c) for c in cluster.channels),
                "iou": ev.mean_iou,
                "dice": ev.mean_dice,
                "delta_iou": ev.mean_iou - baseline_iou,
                "delta_dice": ev.mean_dice - baseline_dice,
            }
        )
    return rows


def write_rows(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cluster_ids_for_focus(
    clusters: list[ChannelCluster],
    ablation: dict[int, dict[str, float]],
    focus_channels: Iterable[int],
    top_k: int = 8,
) -> list[int]:
    lookup = cluster_lookup(clusters)
    selected = {lookup[ch] for ch in focus_channels if ch in lookup}
    scored: list[tuple[float, int]] = []
    for cluster in clusters:
        deltas = [
            abs(ablation[channel].get("delta_iou", 0.0))
            for channel in cluster.channels
            if channel in ablation
        ]
        score = float(np.mean(deltas)) if deltas else 0.0
        scored.append((score, cluster.cluster_id))
    for _score, cluster_id in sorted(scored, reverse=True)[:top_k]:
        selected.add(cluster_id)
    return sorted(selected)
