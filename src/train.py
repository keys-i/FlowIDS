"""Train M0 models."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.config import Config
from src.eval import Batch, Model, evaluate, forward
from src.metrics import binary


def fit(
    model: Model,
    train_loader: DataLoader[Batch],
    validation_loader: DataLoader[Batch],
    config: Config,
    device: torch.device,
) -> tuple[list[dict[str, float | int | None]], dict[str, Tensor]]:
    """Train M0 and return its history with the best CPU checkpoint."""
    train = config.train
    if (
        train.epochs <= 0
        or train.learning_rate <= 0
        or train.weight_decay < 0
        or not 0 <= train.warmup_fraction < 1
        or train.gradient_clip <= 0
        or train.patience <= 0
    ):
        raise ValueError("invalid train configuration")
    if not len(train_loader) or not len(validation_loader):
        raise ValueError("train and validation loaders cannot be empty")

    _ = model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=train.learning_rate,
        weight_decay=train.weight_decay,
    )
    total_steps = train.epochs * len(train_loader)
    warmup_steps = round(total_steps * train.warmup_fraction)
    scheduler = LambdaLR(
        optimizer,
        lambda step: (
            (step + 1) / max(warmup_steps, 1)
            if step < warmup_steps
            else 0.5
            * (1 + math.cos(math.pi * (step - warmup_steps) / max(total_steps - warmup_steps, 1)))
        ),
    )
    amp = bool(train.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_score = float("-inf")
    stale_epochs = 0
    best_state: dict[str, Tensor] = {}
    history: list[dict[str, float | int | None]] = []

    for epoch in range(1, train.epochs + 1):
        _ = model.train()
        loss_total = 0.0
        count = 0
        for batch in train_loader:
            batch = {
                name: value.to(device, non_blocking=device.type == "cuda")
                for name, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = forward(model, batch)
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
            loss_total += loss.detach().item() * len(target)
            count += len(target)

        validation_loss, labels, probabilities = evaluate(model, validation_loader, device)
        validation_auprc = binary(labels, probabilities)["auprc"]
        score = validation_auprc if validation_auprc is not None else -validation_loss
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_total / count,
                "validation_loss": validation_loss,
                "validation_auprc": validation_auprc,
            }
        )
        print(
            f"epoch {epoch}/{train.epochs}",
            f"train_loss={loss_total / count:.6f}",
            f"validation_loss={validation_loss:.6f}",
            f"validation_auprc={validation_auprc}",
        )
        if score > best_score:
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

    return history, best_state
