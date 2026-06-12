"""Train and score the M0 baseline with PyTorch"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from time import perf_counter

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.config import Config
from src.m0.network import FlowTransformer
from src.metrics import binary

type Batch = dict[str, Tensor]


class TimeLimit(TimeoutError):
    """The current phase has used its time budget"""


def batches(loader: Iterable[Batch], deadline: float) -> Iterator[Batch]:
    """Stop between batches at a ``perf_counter`` deadline"""
    if perf_counter() >= deadline:
        raise TimeLimit("time budget exhausted")
    for batch in loader:
        if perf_counter() >= deadline:
            raise TimeLimit("time budget exhausted")
        yield batch


def move(batch: Batch, device: torch.device) -> Batch:
    """Move batch tensors to the device, allowing asynchronous copies on CUDA"""
    return {
        name: value.to(device, non_blocking=device.type == "cuda") for name, value in batch.items()
    }


def evaluate(
    model: FlowTransformer,
    loader: DataLoader[Batch],
    device: torch.device,
    *,
    deadline: float = math.inf,
) -> tuple[float, Tensor, Tensor]:
    """Return mean BCE loss, CPU labels, and CPU attack probabilities

    Args:
        model: Network already on the selected device
        loader: Nonempty labeled batches
        device: Inference device
        deadline: Raise TimeLimit if the full evaluation cannot finish in time

    Restores the original training mode; holds predictions on-device until done
    """
    if not len(loader):
        raise ValueError("evaluation loader cannot be empty")
    training = model.training
    _ = model.eval()
    loss_total = torch.zeros((), device=device)
    labels: list[Tensor] = []
    probabilities: list[Tensor] = []
    try:
        with torch.inference_mode():
            for batch in batches(loader, deadline):
                batch = move(batch, device)
                logits = model(
                    batch["numeric"], batch["missing"], batch["categorical"], batch["padding"]
                )
                target = batch["label"].float()
                loss_total += functional.binary_cross_entropy_with_logits(
                    logits, target, reduction="sum"
                )
                labels.append(target)
                probabilities.append(logits.sigmoid())
            targets = torch.cat(labels).cpu()
            scores = torch.cat(probabilities).cpu()
            return float(loss_total) / len(targets), targets, scores
    finally:
        _ = model.train(training)


def setup_optimizer(model: nn.Module, settings: Config, steps: int) -> tuple[AdamW, LambdaLR]:
    """Create AdamW with linear warmup and cosine decay

    Args:
        model: Network whose trainable parameters will be updated
        settings: Learning rate, decay, warmup, clipping, and patience settings
        steps: Planned optimizer updates
    """
    if (
        steps < 1
        or settings.learning_rate <= 0
        or settings.weight_decay < 0
        or not 0 <= settings.warmup_fraction < 1
        or settings.gradient_clip <= 0
        or settings.patience < 1
    ):
        raise ValueError("invalid optimizer settings")
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    warmup = round(steps * settings.warmup_fraction)
    scheduler = LambdaLR(
        optimizer,
        lambda step: (
            (step + 1) / max(warmup, 1)
            if step < warmup
            else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(steps - warmup, 1)))
        ),
    )
    return optimizer, scheduler


def fit(
    model: FlowTransformer,
    train_loader: DataLoader[Batch],
    validation_loader: DataLoader[Batch],
    config: Config,
    device: torch.device,
    *,
    deadline: float = math.inf,
) -> tuple[list[dict[str, float | int | None]], dict[str, Tensor]]:
    """Train with labels and select weights by validation average precision

    Args:
        model: Network to train
        train_loader: Labeled training batches
        validation_loader: Labeled validation batches
        config: Optimizer, AMP, epoch, and early-stopping settings
        device: Training device
        deadline: Phase cutoff, with the last fifth reserved for validation

    Returns:
        Epoch history and a CPU copy of the best weights. The model keeps its
        last weights; if no validation finishes, return the latest trained weights
    """
    train = config.train
    if train.epochs < 1 or not len(train_loader) or not len(validation_loader):
        raise ValueError("positive epochs and nonempty train/validation loaders are required")
    _ = model.to(device)
    optimizer, scheduler = setup_optimizer(model, train, train.epochs * len(train_loader))
    amp = bool(train.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_score = float("-inf")
    stale_epochs = 0
    best_state: dict[str, Tensor] = {}
    history: list[dict[str, float | int | None]] = []
    train_deadline = perf_counter() + (deadline - perf_counter()) * 0.8

    for epoch in range(1, train.epochs + 1):
        if perf_counter() >= train_deadline:
            break
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        _ = model.train()
        loss_total = torch.zeros((), device=device)
        count = 0
        try:
            for batch in batches(train_loader, train_deadline):
                batch = move(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits = model(
                        batch["numeric"], batch["missing"], batch["categorical"], batch["padding"]
                    )
                    target = batch["label"].float()
                    loss = functional.binary_cross_entropy_with_logits(logits, target)
                if not torch.isfinite(loss):
                    raise ValueError("training loss is not finite")
                scaler.scale(loss).backward()  # pyright: ignore[reportUnusedCallResult]
                _ = scaler.unscale_(optimizer)
                _ = torch.nn.utils.clip_grad_norm_(model.parameters(), train.gradient_clip)
                _ = scaler.step(optimizer)
                _ = scaler.update()
                scheduler.step()
                loss_total += loss.detach() * len(target)
                count += len(target)
        except TimeLimit:
            print("Classification time used; finishing validation")
        if not count:
            break

        train_loss = float(loss_total) / count
        train_seconds = perf_counter() - started
        started = perf_counter()
        validation_loss = validation_auprc = None
        try:
            validation_loss, labels, probabilities = evaluate(
                model, validation_loader, device, deadline=deadline
            )
            validation_auprc = binary(labels, probabilities)["auprc"]
        except TimeLimit:
            print("Validation did not finish; its partial scores are discarded")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_targets": count,
                "validation_loss": validation_loss,
                "validation_auprc": validation_auprc,
                "train_seconds": train_seconds,
                "validation_seconds": perf_counter() - started,
                "train_flows_per_second": count / train_seconds,
                "peak_cuda_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                ),
            }
        )
        print(
            f"epoch {epoch}/{train.epochs}",
            f"train_loss={train_loss:.6f}",
            f"validation_loss={validation_loss}",
            f"validation_auprc={validation_auprc}",
        )
        score = (
            validation_auprc
            if validation_auprc is not None
            else -validation_loss
            if validation_loss is not None
            else float("-inf")
        )
        if score > best_score or not best_state:
            best_score = score
            stale_epochs = 0
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= train.patience:
                print(f"early stop: no validation improvement for {train.patience} epochs")
                break
        if validation_loss is None:
            break

    if not best_state:
        raise TimeLimit("classification ended before any training batch completed")
    return history, best_state


if __name__ == "__main__":
    from unittest.mock import patch

    loader = [{"value": torch.ones(1)}] * 2
    assert len(list(batches(loader, math.inf))) == 2
    with patch(f"{__name__}.perf_counter", side_effect=(0, 0, 2)):
        iterator = batches(loader, 1)
        _ = next(iterator)
        try:
            _ = next(iterator)
        except TimeLimit:
            pass
        else:
            raise AssertionError("batches must stop when time runs out")
    try:
        _ = next(batches(loader, 0))
    except TimeLimit:
        pass
    else:
        raise AssertionError("expired batches must stop")
    print("Batch time limits passed")
