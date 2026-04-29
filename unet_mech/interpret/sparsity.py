"""Activation sparsity scoring: quantify how spatially specialised each bottleneck channel is."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader

from unet_mech.interpret.hooks import get_activations, register_hooks, remove_hooks


def _gini(arr: np.ndarray) -> float:
    """Gini coefficient of a 1-D array (0 = uniform, 1 = maximally sparse)."""
    a = np.sort(np.abs(arr).ravel())
    n = len(a)
    if n == 0 or a.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * (index * a).sum() / (n * a.sum())) - (n + 1) / n)


@torch.no_grad()
def compute_sparsity_scores(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    layer_name: str = "enc_layer4",
    csv_path: str | Path | None = None,
) -> List[dict]:
    """
    For each bottleneck channel, compute:
      - mean activation inside / outside the ground-truth lung mask
      - selectivity index: |in - out| / (in + out)
      - Gini sparsity of the spatial activation map
    Averages across all samples in the loader.
    """
    model.eval()
    register_hooks(model)

    # accumulators per channel: [mean_lung, mean_non_lung, gini]
    accum = None
    n_samples = 0

    try:
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            _ = model(images)
            acts = get_activations(model)

            if layer_name not in acts:
                raise KeyError(f"{layer_name!r} not in activations: {list(acts.keys())}")

            act = acts[layer_name]  # [B, C, h, w]
            B, C, h, w = act.shape

            if accum is None:
                accum = np.zeros((C, 3), dtype=np.float64)

            # downsample mask to activation resolution
            masks_down = F.interpolate(
                masks.float(), size=(h, w), mode="nearest"
            )  # [B, 1, h, w]

            for b in range(B):
                m = masks_down[b, 0].numpy() > 0.5  # [h, w] bool
                for c in range(C):
                    a = act[b, c].numpy()  # [h, w]
                    mag = np.abs(a)
                    in_val = mag[m].mean() if m.any() else 0.0
                    out_val = mag[~m].mean() if (~m).any() else 0.0
                    accum[c, 0] += in_val
                    accum[c, 1] += out_val
                    accum[c, 2] += _gini(a)
                n_samples += 1
    finally:
        remove_hooks(model)

    if accum is None or n_samples == 0:
        return []

    accum /= n_samples
    eps = 1e-8
    rows: List[dict] = []
    for c in range(accum.shape[0]):
        ml, mnl, gi = accum[c]
        sel = abs(ml - mnl) / (ml + mnl + eps)
        rows.append({
            "channel": c,
            "mean_lung": round(ml, 6),
            "mean_non_lung": round(mnl, 6),
            "selectivity_index": round(sel, 4),
            "gini_sparsity": round(gi, 4),
            "region_preference": "lung" if ml > mnl else "non_lung",
        })

    if csv_path is not None:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        fields = ["channel", "mean_lung", "mean_non_lung",
                  "selectivity_index", "gini_sparsity", "region_preference"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        logger.info(f"[sparsity] Wrote {csv_path}")

    top = sorted(rows, key=lambda r: r["gini_sparsity"], reverse=True)[:5]
    logger.info("[sparsity] Top-5 most sparse (monosemantic) channels:")
    for r in top:
        logger.info(f"  ch {r['channel']:>3}  gini={r['gini_sparsity']:.4f}  "
                    f"sel={r['selectivity_index']:.4f}  pref={r['region_preference']}")

    return rows
