"""Build train, validation, and test dataloaders from a transformed dataset."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src.config import Config

from .dataset import TransformedDataset


def build_loaders(
    transformed_dataset: TransformedDataset,
    config: Config,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build dataloaders from the already fitted transformed dataset."""
    split_datasets = {
        "train": Subset(transformed_dataset, transformed_dataset.train_indices),
        "val": Subset(transformed_dataset, transformed_dataset.val_indices),
        "test": Subset(transformed_dataset, transformed_dataset.test_indices),
    }

    train_sampler = None
    if config.dataloaders.balance_train and transformed_dataset.train_sample_weights:
        train_sampler = WeightedRandomSampler(
            weights=transformed_dataset.train_sample_weights,
            num_samples=len(transformed_dataset.train_sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(config.dataloaders.seed),
        )

    base_loader_kwargs = {
        "batch_size": config.dataloaders.batch_size,
        "num_workers": config.dataloaders.num_workers,
        "pin_memory": config.dataloaders.pin_memory,
        "persistent_workers": config.dataloaders.num_workers > 0,
    }

    loaders: dict[str, DataLoader] = {}
    for split_name in ("train", "val", "test"):
        loader_kwargs = dict(base_loader_kwargs)
        loader_kwargs["shuffle"] = split_name == "train" and train_sampler is None
        loader_kwargs["drop_last"] = (
            config.dataloaders.drop_last_train if split_name == "train" else False
        )

        if split_name == "train" and train_sampler is not None:
            loader_kwargs["sampler"] = train_sampler

        loaders[split_name] = DataLoader(
            split_datasets[split_name],
            **loader_kwargs,
        )

    return loaders["train"], loaders["val"], loaders["test"]
