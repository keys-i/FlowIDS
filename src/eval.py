"""Evaluation and plotting utilities for the FlowTransformer baseline."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

from .train import validate_one_epoch


def _plot_lines(
    path: Path,
    title: str,
    ylabel: str,
    series: Sequence[tuple[str, Sequence[float]]],
    *,
    legend_outside: bool = False,
) -> Path | None:
    """Plot one metric figure when it contains finite values."""
    valid = [
        (label, values)
        for label, values in series
        if values and any(math.isfinite(float(value)) for value in values)
    ]
    if not valid:
        return None

    fig, ax = plt.subplots(figsize=(10, 6) if legend_outside else (8, 5))
    for label, values in valid:
        ax.plot(range(1, len(values) + 1), values, label=label, linewidth=1.75)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if len(valid) > 1:
        if legend_outside:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        else:
            ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight" if legend_outside else None)
    plt.close(fig)
    return path


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool = False,
    loss_weights: Mapping[str, float] | None = None,
    binary_class_weight: torch.Tensor | None = None,
    family_class_weight: torch.Tensor | None = None,
    history: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a model and return losses, metrics, and confusion matrices."""
    metrics = validate_one_epoch(
        model,
        loader,
        device,
        amp=amp,
        loss_weights=loss_weights,
        binary_class_weight=binary_class_weight,
        family_class_weight=family_class_weight,
        stage="evaluate",
        collect_predictions=True,
    )

    family_labels = sorted(set(metrics["family_targets"]) | set(metrics["family_predictions"]))
    metrics["family_confusion_matrix"] = confusion_matrix(
        metrics["family_targets"],
        metrics["family_predictions"],
        labels=family_labels,
    )
    metrics["binary_confusion_matrix"] = confusion_matrix(
        metrics["binary_targets"],
        metrics["binary_predictions"],
        labels=[0, 1],
    )

    grad_norms = []
    if history is not None and "train_grad_norm" in history:
        grad_norms = [
            float(value) for value in history["train_grad_norm"] if math.isfinite(float(value))
        ]

    metrics["avg_train_grad_norm"] = (
        sum(grad_norms) / len(grad_norms) if grad_norms else float("nan")
    )
    metrics["summary"] = {
        "total_loss": metrics["total_loss"],
        "binary_loss": metrics["binary_loss"],
        "family_ce": metrics["family_ce"],
        "num_l1": metrics["num_l1"],
        "num_gnll": metrics["num_gnll"],
        "binary_acc": metrics["binary_acc"],
        "family_acc": metrics["family_acc"],
        "avg_train_grad_norm": metrics["avg_train_grad_norm"],
        "aux_class_ce": metrics["aux_class_ce"],
        "aux_class_acc": metrics["aux_class_acc"],
        "aux_num_l1": metrics["aux_num_l1"],
        "aux_gnll": metrics["aux_gnll"],
        "aux_gaussian_mae": metrics["aux_gaussian_mae"],
    }
    return metrics


def plot_training_curves(
    history: Mapping[str, Any],
    out_dir: str | Path,
) -> dict[str, Path]:
    """Plot the training curves that add clear diagnostic value."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    plot_specs = (
        (
            "total_loss.png",
            "Average Training and Validation Loss",
            "Loss",
            [
                ("train_total_loss", history.get("train_total_loss", [])),
                ("val_total_loss", history.get("val_total_loss", [])),
            ],
        ),
        (
            "family_ce.png",
            "Family Cross-Entropy",
            "Cross-Entropy",
            [
                ("train_family_ce", history.get("train_family_ce", [])),
                ("val_family_ce", history.get("val_family_ce", [])),
            ],
        ),
        (
            "binary_loss.png",
            "Binary BCE Loss",
            "BCE Loss",
            [
                ("train_binary_loss", history.get("train_binary_loss", [])),
                ("val_binary_loss", history.get("val_binary_loss", [])),
            ],
        ),
        (
            "numeric_l1.png",
            "Average Numeric L1 Loss",
            "MAE / L1",
            [
                ("train_num_l1", history.get("train_num_l1", [])),
                ("val_num_l1", history.get("val_num_l1", [])),
            ],
        ),
        (
            "gaussian_nll.png",
            "Average Gaussian NLL",
            "Gaussian NLL",
            [
                ("train_num_gnll", history.get("train_num_gnll", [])),
                ("val_num_gnll", history.get("val_num_gnll", [])),
            ],
        ),
    )

    for filename, title, ylabel, series in plot_specs:
        path = _plot_lines(out_dir / filename, title, ylabel, series)
        if path is not None:
            paths[path.stem] = path

    for key, filename, title, ylabel in (
        (
            "val_aux_num_l1_by_feature",
            "numeric_l1_by_feature.png",
            "Per-Feature Numeric L1",
            "MAE / L1",
        ),
        (
            "val_aux_gnll_by_feature",
            "gaussian_nll_by_feature.png",
            "Per-Feature Gaussian NLL",
            "Gaussian NLL",
        ),
    ):
        feature_history = history.get(key) or history.get(key.replace("val_", "train_"))
        if not feature_history:
            continue
        path = _plot_lines(
            out_dir / filename,
            title,
            ylabel,
            sorted(feature_history.items()),
            legend_outside=True,
        )
        if path is not None:
            paths[path.stem] = path

    return paths


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    out_path: str | Path,
    *,
    labels: Sequence[int] | None = None,
    display_labels: Sequence[str] | None = None,
    normalize: str | None = None,
    title: str = "Confusion Matrix",
) -> np.ndarray:
    """Plot and save a confusion matrix."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=display_labels if display_labels is not None else labels,
    ).plot(ax=ax, colorbar=False, values_format=".2f" if normalize else "d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return matrix


def save_summary(
    path: str | Path,
    history: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    checkpoint_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Save a concise JSON training summary."""
    val_total_loss = history.get("val_total_loss", [])
    best_epoch = None
    if val_total_loss:
        finite = [
            (i + 1, float(v)) for i, v in enumerate(val_total_loss) if math.isfinite(float(v))
        ]
        best_epoch = min(finite, key=lambda item: item[1])[0] if finite else None
    best_index = None if best_epoch is None else best_epoch - 1

    summary = {
        "best_epoch": best_epoch,
        "final_epoch": len(history.get("train_total_loss", [])),
        "best_validation_total_loss": float(history["val_total_loss"][best_index])
        if best_index is not None
        else None,
        "best_validation_family_accuracy": float(history["val_family_acc"][best_index])
        if best_index is not None
        else None,
        "best_validation_binary_accuracy": float(history["val_binary_acc"][best_index])
        if best_index is not None
        else None,
        "best_validation_average_numeric_l1": float(history["val_num_l1"][best_index])
        if best_index is not None
        else None,
        "best_validation_average_gaussian_nll": float(history["val_num_gnll"][best_index])
        if best_index is not None
        else None,
        "per_head_loss_summary": {
            "binary_loss": float(evaluation["binary_loss"]),
            "family_ce": float(evaluation["family_ce"]),
            "aux_class_ce": {
                name: float(value) if math.isfinite(float(value)) else None
                for name, value in evaluation.get("aux_class_ce", {}).items()
            },
            "aux_numeric_l1": {
                name: float(value) if math.isfinite(float(value)) else None
                for name, value in evaluation.get("aux_num_l1", {}).items()
            },
            "aux_gaussian_nll": {
                name: float(value) if math.isfinite(float(value)) else None
                for name, value in evaluation.get("aux_gnll", {}).items()
            },
            "aux_gaussian_mae": {
                name: float(value) if math.isfinite(float(value)) else None
                for name, value in evaluation.get("aux_gaussian_mae", {}).items()
            },
        },
        "per_head_accuracy_summary": {
            "binary_accuracy": float(evaluation["binary_acc"]),
            "family_accuracy": float(evaluation["family_acc"]),
            "aux_class_accuracy": {
                name: float(value) if math.isfinite(float(value)) else None
                for name, value in evaluation.get("aux_class_acc", {}).items()
            },
        },
        "checkpoint_paths": {name: str(value) for name, value in checkpoint_paths.items()},
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary


__all__ = ["evaluate", "plot_training_curves", "plot_confusion_matrix", "save_summary"]
