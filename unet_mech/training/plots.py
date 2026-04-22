from __future__ import annotations

from typing import Dict, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


def plot_training_curves(
    history: Dict[str, List[float]],
    freeze_epochs: int,
    title: str = "ResNet18-UNet — Training Curves",
    save_path: str = "training_curves.png",
) -> None:
    """
    Loss and val IoU/Dice. Shaded region = training phase with frozen encoder
    (not the same as proposal "Phase 1" task-metrics phase).
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))
    best_epoch = int(np.argmax(history["val_iou"])) + 1

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    PHASE1_COLOR = "#E8F0FE"
    TRAIN_COLOR = "#2563EB"
    VAL_COLOR = "#DC2626"
    IOU_COLOR = "#16A34A"
    DICE_COLOR = "#9333EA"
    PHASE_LINE = "#64748B"
    BEST_COLOR = "#F59E0B"

    fig, (ax_loss, ax_metrics) = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    def _add_phase_annotation(ax, y_top):
        ax.axvspan(
            0.5,
            freeze_epochs + 0.5,
            color=PHASE1_COLOR,
            alpha=0.6,
            zorder=0,
        )
        ax.axvline(
            freeze_epochs + 0.5, color=PHASE_LINE, linestyle="--", linewidth=1.2, zorder=2
        )
        ax.text(
            freeze_epochs + 0.7,
            y_top * 0.97,
            "encoder\nunfrozen",
            fontsize=7.5,
            color=PHASE_LINE,
            va="top",
            linespacing=1.4,
        )
        ax.axvline(
            best_epoch, color=BEST_COLOR, linestyle=":", linewidth=1.5, zorder=2
        )
        ax.text(
            best_epoch + 0.2,
            y_top * 0.97,
            f"best\n(ep {best_epoch})",
            fontsize=7.5,
            color=BEST_COLOR,
            va="top",
            linespacing=1.4,
        )

    ax_loss.plot(
        epochs,
        history["train_loss"],
        color=TRAIN_COLOR,
        linewidth=2,
        marker="o",
        markersize=3,
        label="Train loss",
        zorder=3,
    )
    ax_loss.plot(
        epochs,
        history["val_loss"],
        color=VAL_COLOR,
        linewidth=2,
        marker="s",
        markersize=3,
        linestyle="--",
        label="Val loss",
        zorder=3,
    )

    y_top_loss = max(max(history["train_loss"]), max(history["val_loss"])) * 1.05
    _add_phase_annotation(ax_loss, y_top_loss)

    ax_loss.set_xlim(0.5, len(epochs) + 0.5)
    ax_loss.set_ylim(bottom=0)
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("DiceBCE Loss", fontsize=11)
    ax_loss.set_title("Loss", fontsize=12)
    phase_patch = mpatches.Patch(
        color=PHASE1_COLOR, alpha=0.6, label="Encoder frozen (initial)"
    )
    ax_loss.legend(
        handles=[
            phase_patch,
            plt.Line2D(
                [],
                [],
                color=TRAIN_COLOR,
                linewidth=2,
                marker="o",
                markersize=4,
                label="Train loss",
            ),
            plt.Line2D(
                [],
                [],
                color=VAL_COLOR,
                linewidth=2,
                marker="s",
                markersize=4,
                linestyle="--",
                label="Val loss",
            ),
        ],
        fontsize=9,
        loc="upper right",
    )

    ax_metrics.plot(
        epochs,
        history["val_iou"],
        color=IOU_COLOR,
        linewidth=2,
        marker="o",
        markersize=3,
        label="Val IoU",
        zorder=3,
    )
    ax_metrics.plot(
        epochs,
        history["val_dice"],
        color=DICE_COLOR,
        linewidth=2,
        marker="s",
        markersize=3,
        linestyle="--",
        label="Val Dice",
        zorder=3,
    )

    best_iou_val = history["val_iou"][best_epoch - 1]
    ax_metrics.scatter(
        [best_epoch],
        [best_iou_val],
        color=BEST_COLOR,
        s=80,
        zorder=5,
        label=f"Best IoU ({best_iou_val:.3f})",
    )

    y_top_metrics = 1.0
    _add_phase_annotation(ax_metrics, y_top_metrics)

    ax_metrics.set_xlim(0.5, len(epochs) + 0.5)
    ax_metrics.set_ylim(0, 1.05)
    ax_metrics.set_xlabel("Epoch", fontsize=11)
    ax_metrics.set_ylabel("Score", fontsize=11)
    ax_metrics.set_title("Validation Metrics", fontsize=12)
    phase_patch2 = mpatches.Patch(
        color=PHASE1_COLOR, alpha=0.6, label="Encoder frozen (initial)"
    )
    ax_metrics.legend(
        handles=[
            phase_patch2,
            plt.Line2D(
                [],
                [],
                color=IOU_COLOR,
                linewidth=2,
                marker="o",
                markersize=4,
                label="Val IoU",
            ),
            plt.Line2D(
                [],
                [],
                color=DICE_COLOR,
                linewidth=2,
                marker="s",
                markersize=4,
                linestyle="--",
                label="Val Dice",
            ),
            plt.Line2D(
                [],
                [],
                color=BEST_COLOR,
                linewidth=0,
                marker="o",
                markersize=7,
                label=f"Best IoU ({best_iou_val:.3f})",
            ),
        ],
        fontsize=9,
        loc="lower right",
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[plot] Training curves saved → {save_path}")
