"""Compute M0 binary classification and calibration metrics"""

from __future__ import annotations

import torch
from torch import Tensor


def binary(labels: Tensor, probability: Tensor, bins: int = 15) -> dict[str, float | int | None]:
    """Compute binary ranking, calibration, and low-FPR metrics

    Args:
        labels: Nonempty binary tensor shaped (n,)
        probability: Finite floating-point scores in [0, 1], on the same device
        bins: Positive number of equal-width ECE bins

    Returns:
        Average precision, tie-correct AUROC, TPR at fixed FPRs, NLL, Brier,
        ECE, and class counts. Average precision uses the empirical step sum
        TPR is None if a class is absent or the budget permits less than one
        observed false positive; AUROC/AP are None when their classes are absent

    Raises:
        ValueError: If shapes, devices, values, or bin count are invalid
    """
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
    ends = torch.nonzero(
        torch.cat((score[1:] != score[:-1], torch.ones(1, device=score.device, dtype=torch.bool)))
    ).flatten()
    positions = ends + 1
    true_positive = target.cumsum(0)[ends]
    false_positive = positions - true_positive
    group_positive = torch.diff(
        true_positive, prepend=torch.zeros(1, device=score.device, dtype=torch.int64)
    )
    group_negative = torch.diff(
        false_positive, prepend=torch.zeros(1, device=score.device, dtype=torch.int64)
    )
    previous_false_positive = false_positive - group_negative
    tp = true_positive.to(torch.float64)
    fp = false_positive.to(torch.float64)
    group_pos = group_positive.to(torch.float64)
    group_neg = group_negative.to(torch.float64)
    previous_fp = previous_false_positive.to(torch.float64)
    rank = positions.to(torch.float64)
    average_precision = float((group_pos * tp / rank).sum() / positives) if positives else 0.0
    auroc_wins = float((group_pos * (negatives - previous_fp - group_neg / 2)).sum())

    def tpr(limit: float) -> float | None:
        """Return the best empirical TPR within a false-positive-rate limit

        Args:
            limit: Maximum false-positive rate

        Returns:
            The best TPR at a score-tie boundary, or `None` when a class is
            absent or `limit` permits fewer than one observed false positive
        """
        if not positives or not negatives or limit * negatives < 1:
            return None
        eligible = tp[fp <= limit * negatives]
        return float(eligible.max() / positives) if eligible.numel() else 0.0

    epsilon = torch.finfo(probability.dtype).eps
    clipped = probability.clamp(epsilon, 1 - epsilon)
    nll = float(
        -(
            labels.to(probability.dtype) * clipped.log()
            + (1 - labels.to(probability.dtype)) * (1 - clipped).log()
        ).mean()
    )
    brier = float(((probability - labels.to(probability.dtype)) ** 2).mean())
    bin_index = torch.clamp((probability * bins).to(torch.int64), max=bins - 1)
    counts = torch.bincount(bin_index, minlength=bins)
    confidence = torch.bincount(bin_index, weights=probability, minlength=bins)
    accuracy = torch.bincount(bin_index, weights=labels.to(probability.dtype), minlength=bins)
    nonempty = counts > 0
    bin_error = (accuracy[nonempty] - confidence[nonempty]).abs() / counts[nonempty]
    ece = float((counts[nonempty].to(probability.dtype) / len(target) * bin_error).sum())

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


if __name__ == "__main__":
    cases = (
        ([1, 0, 1, 0], [0.9, 0.9, 0.1, 0.1], {"auprc": 0.5, "auroc": 0.5}),
        ([1, 0], [0.5, 0.5], {"auprc": 0.5, "auroc": 0.5}),
        ([1, 0], [0.9, 0.1], {"auprc": 1.0, "auroc": 1.0}),
        ([1, 0], [0.1, 0.9], {"auprc": 0.5, "auroc": 0.0}),
        ([0, 0], [0.1, 0.9], {"auprc": None, "auroc": None}),
        ([1, 1], [0.1, 0.9], {"auprc": 1.0, "auroc": None}),
    )
    for labels, score, expected in cases:
        result = binary(torch.tensor(labels), torch.tensor(score))
        for name, value in expected.items():
            assert result[name] == value, (name, result[name], value)
    try:
        _ = binary(torch.tensor([0]), torch.tensor([1.1]))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid probabilities must fail")
    limited = binary(torch.tensor([1] + [0] * 10_000), torch.tensor([1.0] + [0.0] * 10_000))
    assert limited["tpr_at_fpr_1e-4"] == 1.0
    assert limited["tpr_at_fpr_1e-3"] == 1.0
    unresolved = binary(torch.tensor([1] + [0] * 9_999), torch.tensor([1.0] + [0.0] * 9_999))
    assert unresolved["tpr_at_fpr_1e-4"] is None
