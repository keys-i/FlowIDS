"""Evaluate M0 models."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import DataLoader

from src.model.mlp import MLP
from src.model.network import FlowTransformer
from src.obases import Logistic

type Model = FlowTransformer | MLP | Logistic
type Batch = dict[str, Tensor]


def forward(model: Model, batch: Batch) -> Tensor:
    """Return logits using the common M0 batch fields."""
    if isinstance(model, MLP):
        return model(
            batch["numeric"],
            batch["missing"],
            batch["categorical"],
            batch["elapsed"],
        )
    return model(
        batch["numeric"],
        batch["missing"],
        batch["categorical"],
        batch["elapsed"],
        batch["padding"],
        batch["position"],
        batch["causal"],
    )


def evaluate(
    model: Model,
    loader: DataLoader[Batch],
    device: torch.device,
) -> tuple[float, Tensor, Tensor]:
    """Return mean loss, CPU labels, and CPU probabilities for a loader."""
    if not len(loader):
        raise ValueError("evaluation loader cannot be empty")

    training = model.training
    _ = model.eval()
    loss_total = 0.0
    count = 0
    labels: list[Tensor] = []
    probabilities: list[Tensor] = []
    try:
        with torch.inference_mode():
            for batch in loader:
                batch = {
                    name: value.to(device, non_blocking=device.type == "cuda")
                    for name, value in batch.items()
                }
                logits = forward(model, batch)
                target = batch["label"].float()
                loss_total += functional.binary_cross_entropy_with_logits(
                    logits,
                    target,
                    reduction="sum",
                ).item()
                count += len(target)
                labels.append(target.cpu())
                probabilities.append(logits.sigmoid().cpu())
    finally:
        _ = model.train(training)

    return loss_total / count, torch.cat(labels), torch.cat(probabilities)
