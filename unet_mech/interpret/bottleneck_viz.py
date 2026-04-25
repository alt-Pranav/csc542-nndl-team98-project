from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from unet_mech.interpret.hooks import get_activations, register_hooks, remove_hooks
from unet_mech.viz.qualitative import denormalize_imagenet

Selection = Literal["first", "top_abs_mean"]


def _select_channel_indices(
    act: torch.Tensor, max_channels: int, how: Selection
) -> list[int]:
    """act: [1, C, H, W] on CPU"""
    c = act.shape[1]
    if c <= max_channels:
        return list(range(c))
    if how == "first":
        return list(range(max_channels))
    scores = act.abs().mean(dim=(0, 2, 3))
    _, idx = torch.topk(scores, k=max_channels)
    return idx.cpu().tolist()


def save_bottleneck_channel_grid(
    model: nn.Module,
    image: torch.Tensor,
    device: str,
    save_path: str | Path = "outputs/bottleneck_grid.png",
    layer_name: str = "enc_layer4",
    max_channels: int = 32,
    selection: Selection = "top_abs_mean",
    ncols: int = 8,
) -> None:
    """
    Save a grid of bottleneck channel heatmaps: first row = input X-ray, then
    per-channel activations (upsampled to input resolution) in a regular grid.
    """
    model.eval()
    image = image.to(device)
    if image.dim() == 3:
        image = image.unsqueeze(0)

    register_hooks(model)
    try:
        with torch.no_grad():
            _ = model(image)
        acts = get_activations(model)
    finally:
        remove_hooks(model)

    if layer_name not in acts:
        raise KeyError(f"Layer {layer_name!r} not in activations: {list(acts.keys())}")

    act = acts[layer_name]
    b, c, h, w = act.shape
    if b != 1:
        raise ValueError("Expected batch size 1 for this visualisation")

    _, H, W = image.shape[1:]
    idx_list = _select_channel_indices(act, max_channels, how=selection)
    n = len(idx_list)
    nrows_ch = int(math.ceil(n / ncols))
    nrows = 1 + nrows_ch

    fig = plt.figure(figsize=(2.0 * ncols, 1.2 + 2.0 * nrows_ch))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.35, wspace=0.2)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(denormalize_imagenet(image[0]))
    ax0.set_title("Input (ImageNet denorm)", fontsize=10)
    ax0.axis("off")

    for k, ch in enumerate(idx_list):
        r = 1 + k // ncols
        col = k % ncols
        ch_map = act[0, ch]
        up = F.interpolate(
            ch_map.view(1, 1, h, w).float(),
            size=(H, W),
            mode="bilinear",
            align_corners=True,
        )[0, 0].numpy()
        up = (up - up.min()) / (up.max() - up.min() + 1e-8)
        ax = fig.add_subplot(gs[r, col])
        ax.imshow(up, cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"ch {ch}", fontsize=7)
        ax.axis("off")

    for k in range(n, nrows_ch * ncols):
        r = 1 + k // ncols
        col = k % ncols
        ax = fig.add_subplot(gs[r, col])
        ax.axis("off")

    fig.suptitle(
        f"Bottleneck — {layer_name} ({c} ch total, show {n})",
        fontsize=12,
        y=1.0,
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[interpret] Bottleneck grid saved → {save_path}")


def save_bottleneck_with_overlay(
    model: nn.Module,
    image: torch.Tensor,
    device: str,
    save_path: str | Path = "outputs/bottleneck_overlay.png",
    layer_name: str = "enc_layer4",
    channel: int = 0,
    alpha: float = 0.45,
) -> None:
    """Single-channel heatmap overlaid on the X-ray."""
    model.eval()
    image = image.to(device)
    if image.dim() == 3:
        image = image.unsqueeze(0)

    register_hooks(model)
    try:
        with torch.no_grad():
            _ = model(image)
        acts = get_activations(model)
    finally:
        remove_hooks(model)

    act = acts[layer_name]
    c = act.shape[1]
    _, H, W = image.shape[1:]
    if channel < 0 or channel >= c:
        raise ValueError(f"channel {channel} out of range [0, {c - 1}]")

    up = F.interpolate(
        act[:, channel : channel + 1].float(),
        size=(H, W),
        mode="bilinear",
        align_corners=True,
    )[0, 0].numpy()
    up = (up - up.min()) / (up.max() - up.min() + 1e-8)
    ctx = denormalize_imagenet(image[0]) / 255.0

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(ctx)
    ax.imshow(up, cmap="magma", alpha=alpha, vmin=0, vmax=1)
    ax.set_title(f"{layer_name}  channel {channel}")
    ax.axis("off")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[interpret] Overlay saved → {save_path}")
