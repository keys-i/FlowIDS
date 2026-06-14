"""Pretrain M1/M2 on training flows and select weights on masked validation loss"""

import json
import math
from pathlib import Path
from time import perf_counter
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.config import Config
from src.m1.masking import sample_mask
from src.m1.network import Pretrainer
from src.m2.network import M2Pretrainer
from src.train import Batch, move, setup_optimizer


def spread(states: Tensor) -> dict[str, float]:
    """Measure representation spread and covariance rank on CPU

    Args:
        states: At least two event vectors shaped ``(events, width)``

    Returns:
        Median dimension standard deviation, fraction below 1e-3, active rank
        above 1% of the largest covariance eigenvalue, and entropy effective rank
    """
    value = states.float().cpu()
    if value.ndim != 2 or len(value) < 2:
        raise ValueError("spread needs at least two event vectors")
    centered = value - value.mean(0)
    std = value.std(0)
    eigenvalues = cast(Tensor, torch.linalg.eigvalsh(centered.T @ centered / (len(value) - 1)))
    eigenvalues = eigenvalues.clamp_min(0)
    probability = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    effective = torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum())
    return {
        "median_std": float(std.median()),
        "low_std_fraction": float((std < 1e-3).float().mean()),
        "active_rank": float((eigenvalues > eigenvalues.max().clamp_min(1e-12) * 0.01).sum()),
        "effective_rank": float(effective) if eigenvalues.sum() > 0 else 0.0,
    }


@torch.no_grad()
def validate(
    model: Pretrainer, loader: DataLoader[Batch], config: Config, device: torch.device
) -> tuple[float, dict[str, float]]:
    """Score fixed validation masks and sample vectors for collapse diagnostics

    Args:
        model: Pretrainer on the selected device
        loader: Unlabeled validation histories, never final test flows
        config: Masking settings and run seed
        device: Inference device

    Returns:
        Example-weighted masked loss and, for teacher models, teacher/student diagnostics
        from at most 4096 evenly spaced batches' final event vectors
    """
    training = model.training
    _ = model.eval()
    loss_total = torch.zeros((), device=device)
    count = 0
    students: list[Tensor] = []
    teachers: list[Tensor] = []
    generator = torch.Generator().manual_seed(config.run.seed + 1)
    try:
        # Fixed validation masks must not advance the training random stream
        with torch.random.fork_rng(devices=[]):
            for index, batch in enumerate(loader):
                batch = move(batch, device)
                mask = sample_mask(
                    batch["padding"].cpu(),
                    config.pretrain.mask_fraction,
                    config.pretrain.span,
                    generator=generator,
                ).to(device)
                loss, student, teacher = model(batch, mask)
                loss_total += loss * len(mask)
                count += len(mask)
                if teacher is not None and index % max(math.ceil(len(loader) / 16), 1) == 0:
                    students.append(student[:256, -1].cpu())
                    teachers.append(teacher[:256, -1].cpu())
        if not count:
            raise ValueError("pretraining validation loader cannot be empty")
        if not torch.isfinite(loss_total):
            raise ValueError("pretraining validation loss is not finite")
        diagnostics = {
            f"{name}_{metric}": value
            for name, vectors in (("student", students), ("teacher", teachers))
            if vectors
            for metric, value in spread(torch.cat(vectors)).items()
        }
        return float(loss_total) / count, diagnostics
    finally:
        _ = model.train(training)


@torch.no_grad()
def balance_losses(
    model: M2Pretrainer, loader: DataLoader[Batch], config: Config, device: torch.device
) -> dict[str, float | int]:
    """Set inverse-median weights from 200 training-only forward batches

    No optimizer or teacher updates run here. Small loaders repeat without caching
    batches; the weights sum to two and stay fixed for the rest of pretraining
    """
    if not len(loader):
        raise ValueError("loss balancing requires a nonempty training loader")
    training = model.training
    _ = model.eval()
    started = perf_counter()
    values: list[Tensor] = []
    exposures = future_exposures = 0
    generator = torch.Generator().manual_seed(config.run.seed)
    try:
        with torch.random.fork_rng(devices=[]):
            batches = iter(loader)
            for _ in range(200):
                try:
                    batch = next(batches)
                except StopIteration:
                    batches = iter(loader)
                    batch = next(batches)
                batch = move(batch, device)
                mask = sample_mask(
                    batch["padding"].cpu(),
                    config.pretrain.mask_fraction,
                    config.pretrain.span,
                    generator=generator,
                ).to(device)
                losses, _, _ = model.losses(batch, mask)
                values.append(losses.cpu())
                exposures += int((~batch["padding"]).sum())
                if "future_padding" in batch:
                    future_exposures += int((~batch["future_padding"]).sum())
        medians = torch.stack(values).quantile(0.5, dim=0)
        if not torch.isfinite(medians).all() or (medians <= 0).any():
            raise ValueError("loss warm-up needs finite positive median losses")
        weights = medians.reciprocal()
        _ = model.loss_weights.copy_((2 * weights / weights.sum()).to(device))
        return {
            "warmup_batches": 200,
            "warmup_seconds": perf_counter() - started,
            "warmup_exposures": exposures,
            "warmup_future_exposures": future_exposures,
            **{
                f"weight_{name}": float(weight)
                for name, weight in zip(model.loss_names, model.loss_weights, strict=True)
            },
        }
    finally:
        _ = model.train(training)


def pretrain(
    model: Pretrainer,
    train_loader: DataLoader[Batch],
    validation_loader: DataLoader[Batch],
    config: Config,
    device: torch.device,
) -> list[dict[str, float | int]]:
    """Pretrain M1/M2 and restore the best noncollapsed validation weights

    Args:
        model: Student and prediction components
        train_loader: Unlabeled training histories
        validation_loader: Unlabeled validation histories
        config: Pretraining and shared optimizer settings
        device: Training device

    Returns:
        Loss, exposure, timing, and spread history

    Raises:
        ValueError: On invalid settings, non-finite loss, or three collapsed validations
    """
    pre, train = config.pretrain, config.train
    if (
        pre.epochs < 1
        or not len(train_loader)
        or not len(validation_loader)
        or not 0 <= pre.ema_start <= pre.ema_end <= 1
    ):
        raise ValueError("invalid pretraining settings or empty loader")
    _ = model.to(device)
    steps = pre.epochs * len(train_loader)
    optimizer, scheduler = setup_optimizer(model, train, steps)
    amp = bool(train.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    calibration = (
        balance_losses(model, train_loader, config, device)
        if isinstance(model, M2Pretrainer) and len(model.loss_names) > 1
        else {}
    )
    history: list[dict[str, float | int]] = []
    best_state: dict[str, Tensor] = {}
    best_loss = float("inf")
    stale = collapsed = updates = exposures = future_exposures = 0
    momentum = pre.ema_start
    for epoch in range(1, pre.epochs + 1):
        _ = model.train()
        started = perf_counter()
        loss_total = torch.zeros((), device=device)
        count = 0
        for batch in train_loader:
            batch = move(batch, device)
            mask = sample_mask(batch["padding"], pre.mask_fraction, pre.span)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss, _, _ = model(batch, mask)
            if not torch.isfinite(loss):
                raise ValueError("pretraining loss is not finite")
            scaler.scale(loss).backward()
            _ = scaler.unscale_(optimizer)
            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), train.gradient_clip)
            previous_scale = scaler.get_scale()
            _ = scaler.step(optimizer)
            _ = scaler.update()
            if scaler.get_scale() >= previous_scale:
                momentum = (
                    pre.ema_start
                    + (pre.ema_end - pre.ema_start)
                    * (1 - math.cos(math.pi * updates / max(steps - 1, 1)))
                    / 2
                )
                model.update_teacher(momentum)
                scheduler.step()
                updates += 1
            loss_total += loss.detach() * len(mask)
            count += len(mask)
            exposures += int((~batch["padding"]).sum())
            if "future_padding" in batch:
                future_exposures += int((~batch["future_padding"]).sum())
        train_loss = float(loss_total) / count
        seconds = perf_counter() - started
        validation_loss, diagnostics = validate(model, validation_loader, config, device)
        entry: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_seconds": seconds,
            "targets_per_second": count / seconds,
            "exposures": exposures,
            "future_teacher_exposures": future_exposures,
            "updates": updates,
            "ema_momentum": momentum,
            **diagnostics,
            **calibration,
        }
        width = model.encoder.record.numeric.out_features
        bad = any(
            diagnostics.get(f"{name}_median_std", 1) < 1e-3
            or diagnostics.get(f"{name}_low_std_fraction", 0) > 0.5
            or diagnostics.get(f"{name}_active_rank", width) < width * 0.1
            or diagnostics.get(f"{name}_effective_rank", width) < width * 0.1
            or (
                bool(history)
                and validation_loss < history[-1]["validation_loss"]
                and diagnostics.get(f"{name}_effective_rank", width)
                < history[-1].get(f"{name}_effective_rank", width) * 0.5
            )
            for name in ("student", "teacher")
        )
        collapsed = collapsed + 1 if bad else 0
        history.append(entry)
        print(f"pretrain {model.objective} {epoch}/{pre.epochs}: loss={validation_loss:.6f}")
        if collapsed >= 3:
            path = Path(config.run.output)
            path.mkdir(parents=True, exist_ok=True)
            _ = (path / "pretrain_history.json").write_text(json.dumps(history, indent=2) + "\n")
            raise ValueError(
                f"{model.objective} collapsed for three validations; fine-tuning was stopped"
            )
        if validation_loss < best_loss and not bad:
            best_loss, stale = validation_loss, 0
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= train.patience and collapsed == 0:
                break
    if not best_state:
        raise ValueError("pretraining produced no noncollapsed checkpoint")
    _ = model.load_state_dict(best_state)
    return history
