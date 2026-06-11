"""Compute M0 evaluation metrics."""

from __future__ import annotations

import torch
from torch import Tensor


def binary(labels: Tensor, probability: Tensor, bins: int = 15) -> dict[str, float | int | None]:
    """Return empirical binary classification and calibration metrics."""
    if (
        labels.ndim != 1
        or probability.ndim != 1
        or labels.shape != probability.shape
        or labels.device != probability.device
    ):
        raise ValueError("labels and probability must be equal-length 1-D tensors")
    if not labels.numel() or bins <= 0:
        raise ValueError("inputs must be nonempty and bins must be positive")
    if not torch.isfinite(labels).all() or not torch.isfinite(probability).all():
        raise ValueError("labels and probability must be finite")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("labels must be binary")
    if not probability.is_floating_point() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("probability must be floating point values in [0, 1]")

    ranked = torch.argsort(probability, descending=True, stable=True)
    score = probability[ranked]
    target = labels[ranked].to(torch.int64)
    positives = int(target.sum())
    negatives = len(target) - positives
    starts = [
        int(index)
        for index in (
            torch.cat(
                (torch.ones(1, device=score.device, dtype=torch.bool), score[1:] != score[:-1])
            )
            .nonzero()
            .flatten()
        )
    ]
    starts.append(len(target))

    true_positive = false_positive = 0
    average_precision = 0.0
    auroc_wins = 0.0
    points: list[tuple[int, int]] = []
    for start, end in zip(starts, starts[1:]):
        group_positive = int(target[start:end].sum())
        group_negative = end - start - group_positive
        if positives:
            true_positive += group_positive
            average_precision += group_positive / positives * true_positive / end
        auroc_wins += group_positive * (negatives - false_positive - group_negative / 2)
        false_positive += group_negative
        points.append((true_positive, false_positive))

    def tpr(limit: float) -> float | None:
        """Return the best empirical TPR within a false-positive-rate limit."""
        if not positives or not negatives or limit * negatives < 1:
            return None
        return max((tp / positives for tp, fp in points if fp <= limit * negatives), default=0.0)

    epsilon = torch.finfo(probability.dtype).eps
    clipped = probability.clamp(epsilon, 1 - epsilon)
    nll = float(
        -(
            labels.to(probability.dtype) * clipped.log()
            + (1 - labels.to(probability.dtype)) * (1 - clipped).log()
        ).mean()
    )
    brier = float(((probability - labels.to(probability.dtype)) ** 2).mean())
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        in_bin = (probability >= lower) & (
            probability <= (index + 1) / bins
            if index == bins - 1
            else probability < (index + 1) / bins
        )
        if in_bin.any():
            confidence = float(probability[in_bin].mean())
            accuracy = float(labels[in_bin].to(probability.dtype).mean())
            ece += float(in_bin.to(probability.dtype).mean()) * abs(accuracy - confidence)

    return {
        "auprc": average_precision if positives else None,
        "auroc": auroc_wins / (positives * negatives) if positives and negatives else None,
        "tpr_at_fpr_1e-4": tpr(1e-4),
        "tpr_at_fpr_1e-3": tpr(1e-3),
        "tpr_at_fpr_1e-2": tpr(1e-2),
        "tpr_at_1_per_million": tpr(1 / 1_000_000),
        "tpr_at_10_per_million": tpr(10 / 1_000_000),
        "tpr_at_100_per_million": tpr(100 / 1_000_000),
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "positives": positives,
        "negatives": negatives,
    }
