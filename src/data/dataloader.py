"""DataLoader construction and train-only preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.config import Config

from .transform import CategoricalTransform, NumericTransform, TransformedDataset, split


def _collect_train_arrays(
    dataset: Dataset,
    train_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect train-only arrays for fitting transforms."""
    x_num_rows = []
    x_cat_rows = []
    y_binary_rows = []
    y_family_rows = []

    for index in train_indices:
        sample = dataset[index]
        mask = sample["mask"].cpu().numpy().astype(bool)

        if not mask.any():
            continue

        x_num_rows.append(sample["x_num"].cpu().numpy()[mask])
        x_cat_rows.append(sample["x_cat"].cpu().numpy()[mask])
        y_binary_rows.append(sample["y_binary"])
        y_family_rows.append(sample["y_family"])

    if not x_num_rows or not x_cat_rows:
        raise ValueError("Training split did not contain any valid timesteps to fit transforms.")

    x_num_train = np.concatenate(x_num_rows, axis=0)
    x_cat_train = np.concatenate(x_cat_rows, axis=0)
    y_binary_train = np.asarray(y_binary_rows)
    y_family_train = np.asarray(y_family_rows)

    return x_num_train, x_cat_train, y_binary_train, y_family_train


def _collect_targets(
    dataset: Dataset,
    indices: list[int],
    y_binary_dtype: np.dtype[Any],
    y_family_dtype: np.dtype[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect window-level targets for a split."""
    y_binary = np.asarray(
        [dataset[i]["y_binary"] for i in indices],
        dtype=y_binary_dtype,
    )
    y_family = np.asarray(
        [dataset[i]["y_family"] for i in indices],
        dtype=y_family_dtype,
    )
    return y_binary, y_family


def _build_sampler(
    y_binary: np.ndarray,
    y_family: np.ndarray,
    seed: int,
) -> tuple[WeightedRandomSampler, torch.Tensor, torch.Tensor]:
    """Build weighted sampler and loss weights from encoded targets."""
    binary_classes = np.unique(y_binary)
    family_classes = np.unique(y_family)

    binary_weights = compute_class_weight(
        class_weight="balanced",
        classes=binary_classes,
        y=y_binary,
    ).astype(np.float32)

    family_weights = compute_class_weight(
        class_weight="balanced",
        classes=family_classes,
        y=y_family,
    ).astype(np.float32)

    binary_map = {
        int(cls): float(weight) for cls, weight in zip(binary_classes, binary_weights, strict=False)
    }
    family_map = {
        int(cls): float(weight) for cls, weight in zip(family_classes, family_weights, strict=False)
    }

    sample_weights = [
        (binary_map[int(yb)] * family_map[int(yf)]) ** 0.5
        for yb, yf in zip(y_binary, y_family, strict=False)
    ]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )

    return (
        sampler,
        torch.tensor(binary_weights, dtype=torch.float32),
        torch.tensor(family_weights, dtype=torch.float32),
    )


def build_loaders(
    dataset: Dataset,
    config: Config,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    """Fit transforms on train only and build train/val/test loaders."""
    split_cfg = config.dataset.split
    transforms_cfg = config.dataset.transforms

    if split_cfg.mode != "temporal":
        raise ValueError(f"Unsupported split mode: {split_cfg.mode}.")

    train_indices, val_indices, test_indices = split(
        dataset=dataset,
        splits=split_cfg.splits,
        purge=split_cfg.purge,
    )

    x_num_train, x_cat_train, y_binary_train, y_family_train = _collect_train_arrays(
        dataset=dataset,
        train_indices=train_indices,
    )

    numeric_transform = NumericTransform(
        clip_quantiles=transforms_cfg.clip_quantiles,
    )
    numeric_transform.fit(x_num_train)

    categorical_transform = CategoricalTransform()
    categorical_transform.fit(
        x_cat=x_cat_train,
        y_binary=y_binary_train,
        y_family=y_family_train,
    )

    split_indices = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }

    encoded_targets = {}
    split_datasets = {}

    for split_name, indices in split_indices.items():
        y_binary_raw, y_family_raw = _collect_targets(
            dataset=dataset,
            indices=indices,
            y_binary_dtype=y_binary_train.dtype,
            y_family_dtype=y_family_train.dtype,
        )

        y_binary_encoded, y_family_encoded = categorical_transform.transform_targets(
            y_binary_raw,
            y_family_raw,
        )
        encoded_targets[split_name] = (y_binary_encoded, y_family_encoded)

        split_datasets[split_name] = TransformedDataset(
            dataset=dataset,
            indices=indices,
            numeric_transform=numeric_transform,
            categorical_transform=categorical_transform,
            encoded_binary=y_binary_encoded,
            encoded_family=y_family_encoded,
            train=(split_name == "train"),
            noise_std=transforms_cfg.noise_std if split_name == "train" else 0.0,
        )

    train_binary, train_family = encoded_targets["train"]

    sampler = None
    binary_loss_weight = None
    family_loss_weight = None

    if config.dataloaders.balance_train:
        sampler, binary_loss_weight, family_loss_weight = _build_sampler(
            y_binary=train_binary,
            y_family=train_family,
            seed=config.dataloaders.seed,
        )
    else:
        _, binary_loss_weight, family_loss_weight = _build_sampler(
            y_binary=train_binary,
            y_family=train_family,
            seed=config.dataloaders.seed,
        )

    persistent_workers = config.dataloaders.num_workers > 0

    loader_kwargs = {
        "batch_size": config.dataloaders.batch_size,
        "num_workers": config.dataloaders.num_workers,
        "pin_memory": config.dataloaders.pin_memory,
        "persistent_workers": persistent_workers,
    }

    loaders = {}

    for split_name in ("train", "val", "test"):
        kwargs = dict(loader_kwargs)
        kwargs["shuffle"] = split_name == "train" and sampler is None
        kwargs["drop_last"] = config.dataloaders.drop_last_train if split_name == "train" else False

        if split_name == "train" and sampler is not None:
            kwargs["sampler"] = sampler

        loaders[split_name] = DataLoader(
            split_datasets[split_name],
            **kwargs,
        )

    state = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "numeric_transform": numeric_transform,
        "categorical_transform": categorical_transform,
        "binary_loss_weight": binary_loss_weight,
        "family_loss_weight": family_loss_weight,
    }

    return loaders["train"], loaders["val"], loaders["test"], state
