"""Training loops for the FlowTransformer baseline."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def _build_losses(
    device: torch.device,
    binary_class_weight: torch.Tensor | None,
    family_class_weight: torch.Tensor | None,
) -> tuple[nn.BCEWithLogitsLoss, nn.CrossEntropyLoss, nn.GaussianNLLLoss]:
    """Build the loss modules used during training and evaluation."""
    pos_weight = None
    if binary_class_weight is not None:
        weights = binary_class_weight.to(device=device, dtype=torch.float32).flatten()
        if weights.numel() == 1:
            pos_weight = weights
        elif weights.numel() >= 2 and weights[0] > 0:
            pos_weight = (weights[1] / weights[0]).unsqueeze(0)

    family_weight = None
    if family_class_weight is not None:
        family_weight = family_class_weight.to(device=device, dtype=torch.float32)

    return (
        nn.BCEWithLogitsLoss(pos_weight=pos_weight),
        nn.CrossEntropyLoss(weight=family_weight),
        nn.GaussianNLLLoss(reduction="none"),
    )


def _align_target(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Align a regression target to the prediction shape."""
    if prediction.ndim == target.ndim + 1 and prediction.size(-1) == 1:
        return target.unsqueeze(-1)
    if target.ndim == prediction.ndim + 1 and target.size(-1) == 1:
        return target.squeeze(-1)
    return target


def _empty_stats(
    *,
    collect_predictions: bool = False,
    track_grad: bool = False,
) -> dict[str, Any]:
    """Create one metrics accumulator."""
    stats: dict[str, Any] = {
        "sample_count": 0,
        "total_loss_sum": 0.0,
        "binary_loss_sum": 0.0,
        "family_loss_sum": 0.0,
        "binary_correct": 0,
        "family_correct": 0,
        "num_l1_sum": 0.0,
        "num_l1_count": 0,
        "gnll_sum": 0.0,
        "gnll_count": 0,
        "aux_class_loss_sum": defaultdict(float),
        "aux_class_count": defaultdict(int),
        "aux_class_correct": defaultdict(int),
        "aux_num_loss_sum": defaultdict(float),
        "aux_num_count": defaultdict(int),
        "aux_num_feature_sum": defaultdict(float),
        "aux_num_feature_count": defaultdict(int),
        "aux_gnll_sum": defaultdict(float),
        "aux_gnll_count": defaultdict(int),
        "aux_gnll_feature_sum": defaultdict(float),
        "aux_gnll_feature_count": defaultdict(int),
        "aux_gaussian_mae_sum": defaultdict(float),
        "aux_gaussian_mae_count": defaultdict(int),
    }
    if collect_predictions:
        stats["family_targets"] = []
        stats["family_predictions"] = []
        stats["binary_targets"] = []
        stats["binary_predictions"] = []
    if track_grad:
        stats["grad_norm_sum"] = 0.0
        stats["grad_norm_count"] = 0
    return stats


def _class_terms(
    outputs: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    batch: dict[str, torch.Tensor],
    device: torch.device,
    loss_weights: Mapping[str, float] | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute auxiliary categorical losses and metrics."""
    total = outputs["binary_logits"].new_zeros(())
    stats = _empty_stats()

    for name, logits in outputs.items():
        if name in {"binary_logits", "family_logits"} or not name.endswith("_logits"):
            continue
        head = name.removesuffix("_logits")
        target_key = f"y_{head}"
        if target_key not in batch:
            continue

        target = batch[target_key].to(device=device, dtype=torch.long, non_blocking=True)
        valid = target >= 0
        if not torch.any(valid):
            continue

        logits = logits[valid]
        target = target[valid]
        head_loss = F.cross_entropy(logits, target)
        head_weight = (
            float(loss_weights.get(head, loss_weights.get("aux_class", 1.0)))
            if loss_weights
            else 1.0
        )
        total = total + head_weight * head_loss

        count = int(target.numel())
        stats["aux_class_loss_sum"][head] += head_loss.item() * count
        stats["aux_class_count"][head] += count
        stats["aux_class_correct"][head] += int(logits.argmax(dim=1).eq(target).sum().item())

    return total, stats


def _regression_terms(
    outputs: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    batch: dict[str, torch.Tensor],
    device: torch.device,
    loss_weights: Mapping[str, float] | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute auxiliary L1 regression losses and metrics."""
    total = outputs["binary_logits"].new_zeros(())
    stats = _empty_stats()

    for name, prediction in outputs.items():
        if not name.endswith("_pred"):
            continue
        head = name.removesuffix("_pred")
        target_key = f"y_{head}"
        if target_key not in batch:
            continue

        target = batch[target_key].to(device=device, dtype=prediction.dtype, non_blocking=True)
        target = _align_target(prediction, target)
        valid = torch.isfinite(target)
        if not torch.any(valid):
            continue

        l1 = F.l1_loss(prediction, target, reduction="none")
        head_loss = l1[valid].mean()
        head_weight = (
            float(loss_weights.get(head, loss_weights.get("aux_reg", 1.0))) if loss_weights else 1.0
        )
        total = total + head_weight * head_loss

        count = int(valid.sum().item())
        stats["num_l1_sum"] += l1[valid].sum().item()
        stats["num_l1_count"] += count
        stats["aux_num_loss_sum"][head] += l1[valid].sum().item()
        stats["aux_num_count"][head] += count

        if l1.ndim == 1:
            l1 = l1.unsqueeze(-1)
            valid = valid.unsqueeze(-1)

        for index in range(l1.size(-1)):
            label = head if l1.size(-1) == 1 else f"{head}[{index}]"
            feature_valid = valid[:, index]
            if torch.any(feature_valid):
                stats["aux_num_feature_sum"][label] += l1[:, index][feature_valid].sum().item()
                stats["aux_num_feature_count"][label] += int(feature_valid.sum().item())

    return total, stats


def _gaussian_terms(
    outputs: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    batch: dict[str, torch.Tensor],
    device: torch.device,
    gaussian_criterion: nn.GaussianNLLLoss,
    loss_weights: Mapping[str, float] | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute auxiliary Gaussian NLL losses and metrics."""
    total = outputs["binary_logits"].new_zeros(())
    stats = _empty_stats()

    for name, pair in outputs.items():
        if not name.endswith("_mean_var"):
            continue
        head = name.removesuffix("_mean_var")
        target_key = f"y_{head}"
        if target_key not in batch:
            continue

        mean, variance = pair
        target = batch[target_key].to(device=device, dtype=mean.dtype, non_blocking=True)
        target = _align_target(mean, target)
        valid = torch.isfinite(target)
        if not torch.any(valid):
            continue

        gnll = gaussian_criterion(mean, target, variance)
        mae = (mean - target).abs()
        head_loss = gnll[valid].mean()
        head_weight = (
            float(loss_weights.get(head, loss_weights.get("aux_gaussian", 1.0)))
            if loss_weights
            else 1.0
        )
        total = total + head_weight * head_loss

        count = int(valid.sum().item())
        stats["gnll_sum"] += gnll[valid].sum().item()
        stats["gnll_count"] += count
        stats["aux_gnll_sum"][head] += gnll[valid].sum().item()
        stats["aux_gnll_count"][head] += count
        stats["aux_gaussian_mae_sum"][head] += mae[valid].sum().item()
        stats["aux_gaussian_mae_count"][head] += count

        if gnll.ndim == 1:
            gnll = gnll.unsqueeze(-1)
            valid = valid.unsqueeze(-1)

        for index in range(gnll.size(-1)):
            label = head if gnll.size(-1) == 1 else f"{head}[{index}]"
            feature_valid = valid[:, index]
            if torch.any(feature_valid):
                stats["aux_gnll_feature_sum"][label] += gnll[:, index][feature_valid].sum().item()
                stats["aux_gnll_feature_count"][label] += int(feature_valid.sum().item())

    return total, stats


def _compute_batch(
    outputs: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    batch: dict[str, torch.Tensor],
    binary_criterion: nn.BCEWithLogitsLoss,
    family_criterion: nn.CrossEntropyLoss,
    gaussian_criterion: nn.GaussianNLLLoss,
    loss_weights: Mapping[str, float] | None,
    *,
    collect_predictions: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute one batch loss and all batch-level statistics."""
    device = outputs["binary_logits"].device
    y_binary = batch["y_binary"].to(device=device, dtype=torch.float32, non_blocking=True)
    y_family = batch["y_family"].to(device=device, dtype=torch.long, non_blocking=True)
    batch_size = int(y_binary.size(0))

    binary_weight = float(loss_weights.get("binary", 1.0)) if loss_weights else 1.0
    family_weight = float(loss_weights.get("family", 1.0)) if loss_weights else 1.0
    binary_loss = binary_criterion(outputs["binary_logits"], y_binary)
    family_loss = family_criterion(outputs["family_logits"], y_family)
    total_loss = binary_weight * binary_loss + family_weight * family_loss

    class_loss, class_stats = _class_terms(outputs, batch, device, loss_weights)
    reg_loss, reg_stats = _regression_terms(outputs, batch, device, loss_weights)
    gaussian_loss, gaussian_stats = _gaussian_terms(
        outputs,
        batch,
        device,
        gaussian_criterion,
        loss_weights,
    )
    total_loss = total_loss + class_loss + reg_loss + gaussian_loss

    stats = _empty_stats(collect_predictions=collect_predictions)
    stats["sample_count"] = batch_size
    stats["total_loss_sum"] = total_loss.item() * batch_size
    stats["binary_loss_sum"] = binary_loss.item() * batch_size
    stats["family_loss_sum"] = family_loss.item() * batch_size
    stats["binary_correct"] = int((outputs["binary_logits"] >= 0).eq(y_binary > 0.5).sum().item())
    stats["family_correct"] = int(outputs["family_logits"].argmax(dim=1).eq(y_family).sum().item())

    for source in (class_stats, reg_stats, gaussian_stats):
        _merge_stats(stats, source)

    if collect_predictions:
        stats["family_targets"] = y_family.detach().cpu().tolist()
        stats["family_predictions"] = outputs["family_logits"].argmax(dim=1).detach().cpu().tolist()
        stats["binary_targets"] = y_binary.long().detach().cpu().tolist()
        stats["binary_predictions"] = (outputs["binary_logits"] >= 0).long().detach().cpu().tolist()

    return total_loss, stats


def _merge_stats(running: dict[str, Any], batch_stats: dict[str, Any]) -> None:
    """Accumulate one batch of statistics into the epoch totals."""
    for key in (
        "sample_count",
        "total_loss_sum",
        "binary_loss_sum",
        "family_loss_sum",
        "binary_correct",
        "family_correct",
        "num_l1_sum",
        "num_l1_count",
        "gnll_sum",
        "gnll_count",
    ):
        running[key] += batch_stats[key]

    for key in (
        "aux_class_loss_sum",
        "aux_class_count",
        "aux_class_correct",
        "aux_num_loss_sum",
        "aux_num_count",
        "aux_num_feature_sum",
        "aux_num_feature_count",
        "aux_gnll_sum",
        "aux_gnll_count",
        "aux_gnll_feature_sum",
        "aux_gnll_feature_count",
        "aux_gaussian_mae_sum",
        "aux_gaussian_mae_count",
    ):
        for name, value in batch_stats[key].items():
            running[key][name] += value

    for key in ("family_targets", "family_predictions", "binary_targets", "binary_predictions"):
        if key in batch_stats:
            running[key].extend(batch_stats[key])


def _finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """Convert running sums into epoch averages."""
    mean = lambda total, count: total / count if count else float("nan")
    metrics = {
        "total_loss": mean(stats["total_loss_sum"], stats["sample_count"]),
        "binary_loss": mean(stats["binary_loss_sum"], stats["sample_count"]),
        "family_ce": mean(stats["family_loss_sum"], stats["sample_count"]),
        "num_l1": mean(stats["num_l1_sum"], stats["num_l1_count"]),
        "num_gnll": mean(stats["gnll_sum"], stats["gnll_count"]),
        "binary_acc": mean(float(stats["binary_correct"]), stats["sample_count"]),
        "family_acc": mean(float(stats["family_correct"]), stats["sample_count"]),
        "aux_class_ce": {
            name: mean(stats["aux_class_loss_sum"][name], stats["aux_class_count"][name])
            for name in sorted(stats["aux_class_loss_sum"])
        },
        "aux_class_acc": {
            name: mean(float(stats["aux_class_correct"][name]), stats["aux_class_count"][name])
            for name in sorted(stats["aux_class_correct"])
        },
        "aux_num_l1": {
            name: mean(stats["aux_num_loss_sum"][name], stats["aux_num_count"][name])
            for name in sorted(stats["aux_num_loss_sum"])
        },
        "aux_num_l1_by_feature": {
            name: mean(stats["aux_num_feature_sum"][name], stats["aux_num_feature_count"][name])
            for name in sorted(stats["aux_num_feature_sum"])
        },
        "aux_gnll": {
            name: mean(stats["aux_gnll_sum"][name], stats["aux_gnll_count"][name])
            for name in sorted(stats["aux_gnll_sum"])
        },
        "aux_gnll_by_feature": {
            name: mean(stats["aux_gnll_feature_sum"][name], stats["aux_gnll_feature_count"][name])
            for name in sorted(stats["aux_gnll_feature_sum"])
        },
        "aux_gaussian_mae": {
            name: mean(stats["aux_gaussian_mae_sum"][name], stats["aux_gaussian_mae_count"][name])
            for name in sorted(stats["aux_gaussian_mae_sum"])
        },
    }

    if "grad_norm_sum" in stats:
        metrics["grad_norm"] = mean(stats["grad_norm_sum"], stats["grad_norm_count"])
    for key in ("family_targets", "family_predictions", "binary_targets", "binary_predictions"):
        if key in stats:
            metrics[key] = stats[key]
    return metrics


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    stage: str,
    optimizer: torch.optim.Optimizer | None = None,
    amp: bool = False,
    grad_clip_norm: float | None = None,
    loss_weights: Mapping[str, float] | None = None,
    binary_class_weight: torch.Tensor | None = None,
    family_class_weight: torch.Tensor | None = None,
    epoch: int | None = None,
    epochs: int | None = None,
    collect_predictions: bool = False,
) -> dict[str, Any]:
    """Run one training, validation, or evaluation epoch."""
    binary_criterion, family_criterion, gaussian_criterion = _build_losses(
        device,
        binary_class_weight,
        family_class_weight,
    )
    training = optimizer is not None
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=training and use_amp)
    stats = _empty_stats(collect_predictions=collect_predictions, track_grad=training)

    model.train(training)
    desc = stage if epoch is None or epochs is None else f"{stage} {epoch}/{epochs}"
    progress = tqdm(loader, desc=desc, leave=False)
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch in progress:
            x_num = batch["x_num"].to(device=device, dtype=torch.float32, non_blocking=True)
            x_cat = batch["x_cat"].to(device=device, dtype=torch.long, non_blocking=True)
            mask = batch["mask"].to(device=device, dtype=torch.bool, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(x_num, x_cat, mask)
                total_loss, batch_stats = _compute_batch(
                    outputs,
                    batch,
                    binary_criterion,
                    family_criterion,
                    gaussian_criterion,
                    loss_weights,
                    collect_predictions=collect_predictions,
                )

            if training:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_norm is None:
                    grads = [
                        p.grad.detach().float().norm(2)
                        for p in model.parameters()
                        if p.grad is not None
                    ]
                    grad_norm = torch.stack(grads).norm(2).item() if grads else 0.0
                else:
                    grad_norm = clip_grad_norm_(model.parameters(), grad_clip_norm).item()
                scaler.step(optimizer)
                scaler.update()
                stats["grad_norm_sum"] += grad_norm
                stats["grad_norm_count"] += 1

            _merge_stats(stats, batch_stats)
            postfix = {
                "loss": f"{stats['total_loss_sum'] / stats['sample_count']:.4f}",
                "bin": f"{stats['binary_loss_sum'] / stats['sample_count']:.4f}",
                "fam": f"{stats['family_loss_sum'] / stats['sample_count']:.4f}",
            }
            if stats["num_l1_count"]:
                postfix["l1"] = f"{stats['num_l1_sum'] / stats['num_l1_count']:.4f}"
            if stats["gnll_count"]:
                postfix["gnll"] = f"{stats['gnll_sum'] / stats['gnll_count']:.4f}"
            progress.set_postfix(postfix)

    progress.close()
    return _finalize_stats(stats)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    amp: bool = False,
    grad_clip_norm: float | None = None,
    loss_weights: Mapping[str, float] | None = None,
    binary_class_weight: torch.Tensor | None = None,
    family_class_weight: torch.Tensor | None = None,
    epoch: int | None = None,
    epochs: int | None = None,
) -> dict[str, Any]:
    """Train for one epoch and return epoch averages."""
    return _run_epoch(
        model,
        loader,
        device,
        stage="train",
        optimizer=optimizer,
        amp=amp,
        grad_clip_norm=grad_clip_norm,
        loss_weights=loss_weights,
        binary_class_weight=binary_class_weight,
        family_class_weight=family_class_weight,
        epoch=epoch,
        epochs=epochs,
    )


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool = False,
    loss_weights: Mapping[str, float] | None = None,
    binary_class_weight: torch.Tensor | None = None,
    family_class_weight: torch.Tensor | None = None,
    epoch: int | None = None,
    epochs: int | None = None,
    stage: str = "validate",
    collect_predictions: bool = False,
) -> dict[str, Any]:
    """Validate for one epoch and return epoch averages."""
    return _run_epoch(
        model,
        loader,
        device,
        stage=stage,
        amp=amp,
        loss_weights=loss_weights,
        binary_class_weight=binary_class_weight,
        family_class_weight=family_class_weight,
        epoch=epoch,
        epochs=epochs,
        collect_predictions=collect_predictions,
    )


def _empty_history() -> dict[str, Any]:
    """Create the training history structure."""
    history = {
        key: []
        for key in (
            "train_total_loss",
            "val_total_loss",
            "train_binary_loss",
            "val_binary_loss",
            "train_family_ce",
            "val_family_ce",
            "train_num_l1",
            "val_num_l1",
            "train_num_gnll",
            "val_num_gnll",
            "train_binary_acc",
            "val_binary_acc",
            "train_family_acc",
            "val_family_acc",
            "train_grad_norm",
        )
    }
    for key in (
        "train_aux_class_ce",
        "val_aux_class_ce",
        "train_aux_class_acc",
        "val_aux_class_acc",
        "train_aux_num_l1",
        "val_aux_num_l1",
        "train_aux_num_l1_by_feature",
        "val_aux_num_l1_by_feature",
        "train_aux_gnll",
        "val_aux_gnll",
        "train_aux_gnll_by_feature",
        "val_aux_gnll_by_feature",
        "train_aux_gaussian_mae",
        "val_aux_gaussian_mae",
    ):
        history[key] = {}
    return history


def _append_nested_history(
    history: dict[str, Any],
    prefix: str,
    metrics: dict[str, Any],
    epoch_index: int,
) -> None:
    """Append one epoch of nested per-head history."""
    for name in (
        "aux_class_ce",
        "aux_class_acc",
        "aux_num_l1",
        "aux_num_l1_by_feature",
        "aux_gnll",
        "aux_gnll_by_feature",
        "aux_gaussian_mae",
    ):
        key = f"{prefix}_{name}"
        for head, series in history[key].items():
            if head not in metrics[name]:
                series.append(float("nan"))
        for head, value in metrics[name].items():
            history[key].setdefault(head, [float("nan")] * epoch_index).append(float(value))


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    checkpoint_dir: str | Path,
    *,
    device: torch.device,
    amp: bool = False,
    grad_clip_norm: float | None = None,
    loss_weights: Mapping[str, float] | None = None,
    binary_class_weight: torch.Tensor | None = None,
    family_class_weight: torch.Tensor | None = None,
    scheduler: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the model, save checkpoints, and return history and summary."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = _empty_history()
    best_epoch = -1
    best_val_total_loss = float("inf")
    best_checkpoint_path = checkpoint_dir / "best.pt"
    final_checkpoint_path = checkpoint_dir / "final.pt"
    epoch_bar = tqdm(range(1, epochs + 1), desc="epochs")

    for epoch in epoch_bar:
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            amp=amp,
            grad_clip_norm=grad_clip_norm,
            loss_weights=loss_weights,
            binary_class_weight=binary_class_weight,
            family_class_weight=family_class_weight,
            epoch=epoch,
            epochs=epochs,
        )
        val_metrics = validate_one_epoch(
            model,
            val_loader,
            device,
            amp=amp,
            loss_weights=loss_weights,
            binary_class_weight=binary_class_weight,
            family_class_weight=family_class_weight,
            epoch=epoch,
            epochs=epochs,
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["total_loss"])
            else:
                scheduler.step()

        for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
            history[f"{prefix}_total_loss"].append(metrics["total_loss"])
            history[f"{prefix}_binary_loss"].append(metrics["binary_loss"])
            history[f"{prefix}_family_ce"].append(metrics["family_ce"])
            history[f"{prefix}_num_l1"].append(metrics["num_l1"])
            history[f"{prefix}_num_gnll"].append(metrics["num_gnll"])
            history[f"{prefix}_binary_acc"].append(metrics["binary_acc"])
            history[f"{prefix}_family_acc"].append(metrics["family_acc"])
            _append_nested_history(history, prefix, metrics, epoch - 1)

        history["train_grad_norm"].append(train_metrics["grad_norm"])
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "history": history,
        }

        if val_metrics["total_loss"] < best_val_total_loss:
            best_val_total_loss = float(val_metrics["total_loss"])
            best_epoch = epoch
            torch.save(checkpoint, best_checkpoint_path)
        if epoch == epochs:
            torch.save(checkpoint, final_checkpoint_path)

        epoch_bar.set_postfix(
            train_loss=f"{train_metrics['total_loss']:.4f}",
            val_loss=f"{val_metrics['total_loss']:.4f}",
            bin_acc=f"{val_metrics['binary_acc']:.4f}",
            fam_acc=f"{val_metrics['family_acc']:.4f}",
        )

    epoch_bar.close()
    return history, {
        "best_epoch": best_epoch,
        "final_epoch": epochs,
        "best_val_total_loss": best_val_total_loss,
        "best_checkpoint_path": best_checkpoint_path,
        "final_checkpoint_path": final_checkpoint_path,
    }


__all__ = ["train_one_epoch", "validate_one_epoch", "fit"]
