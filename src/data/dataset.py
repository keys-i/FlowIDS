"""Raw and transformed datasets for the flow pipeline."""

from __future__ import annotations

from typing import Any, Optional, cast

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.config import Config

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .transform import CategoricalTransform, NumericTransform, TargetTransform
from .window_split import build_fixed_windows


class FlowDataset(Dataset):
    """Read raw flow rows, sort them, and expose fixed windows."""

    def __init__(self, config: Config) -> None:
        targets_cfg = config.dataset.targets
        window_cfg = config.dataset.window

        self.numeric_features = NUMERIC_FEATURES
        self.categorical_features = CATEGORICAL_FEATURES
        self.window_len = window_cfg.len
        self.binary_target = targets_cfg.binary
        self.family_target = targets_cfg.family

        columns = (
            [config.dataset.time_col]
            + self.numeric_features
            + self.categorical_features
            + [self.binary_target, self.family_target]
        )

        with tqdm(total=2, desc="dataset", unit="step") as progress:
            progress.set_postfix_str("load parquet")
            self.df = (
                pl.scan_parquet(config.paths.data)
                .select(columns)
                .sort(config.dataset.time_col)
                .collect(engine=config.dataset.polars.engine)
            )
            progress.update()

            progress.set_postfix_str("build windows")
            self.windows = build_fixed_windows(
                nrows=self.df.height,
                window_len=self.window_len,
                stride=window_cfg.stride,
                drop_incomplete=window_cfg.drop_incomplete,
                pad_value=window_cfg.pad_value,
            )
            progress.update()

    def __len__(self) -> int:
        """Return the number of temporal windows."""
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one raw window sample without preprocessing."""
        window_info = self.windows[index]
        start = window_info["start"]
        length = window_info["length"]

        window = self.df.slice(start, length)

        x_num = torch.tensor(
            window.select(self.numeric_features).to_numpy(),
            dtype=torch.float32,
        )
        x_cat = torch.tensor(
            window.select(self.categorical_features).to_numpy(),
            dtype=torch.long,
        )

        if length < self.window_len:
            pad_rows = self.window_len - length
            x_num = F.pad(x_num, pad=(0, 0, 0, pad_rows), value=0.0)
            x_cat = F.pad(x_cat, pad=(0, 0, 0, pad_rows), value=0)

        target_row = window.row(length - 1, named=True)
        return {
            "x_num": x_num,
            "x_cat": x_cat,
            "y_binary": target_row[self.binary_target],
            "y_family": target_row[self.family_target],
            "mask": torch.tensor(window_info["mask"], dtype=torch.bool),
        }


class TransformedDataset(Dataset):
    """Wrap the raw dataset and fit/apply train-only preprocessing."""

    def __init__(self, dataset: FlowDataset, config: Config) -> None:
        self.dataset = dataset
        self.config = config

        self.numeric_transform = NumericTransform(
            clip_quantiles=config.dataset.transforms.clip_quantiles,
        )
        self.categorical_transform = CategoricalTransform()
        self.target_transform = TargetTransform()

        self.train_indices: list[int] = []
        self.val_indices: list[int] = []
        self.test_indices: list[int] = []
        self._train_end = 0

        self.binary_loss_weight: Optional[torch.Tensor] = None
        self.family_loss_weight: Optional[torch.Tensor] = None
        self.train_sample_weights: list[float] = []

        self.fit()

    @property
    def categorical_cardinalities(self) -> dict[str, int]:
        """Return embedding cardinalities for categorical inputs."""
        encoder = self.categorical_transform.encoder_
        if encoder is None:
            raise RuntimeError("CategoricalTransform must be fit before reading cardinalities.")

        return {
            name: len(categories) + 2
            for name, categories in zip(
                self.dataset.categorical_features,
                encoder.categories_,
                strict=True,
            )
        }

    def fit(self) -> "TransformedDataset":
        """Fit all preprocessing on the train split only."""
        split_cfg = self.config.dataset.split
        if split_cfg.mode != "temporal":
            raise ValueError(f"Unsupported split mode: {split_cfg.mode}.")

        n = len(self.dataset)
        train_end = int(n * split_cfg.splits[0])
        val_end = int(n * (split_cfg.splits[0] + split_cfg.splits[1]))

        self.train_indices = list(range(0, train_end))
        self.val_indices = list(range(min(train_end + split_cfg.purge, n), val_end))
        self.test_indices = list(range(min(val_end + split_cfg.purge, n), n))
        self._train_end = train_end

        x_num_rows = []
        x_cat_rows = []
        y_binary_rows = []
        y_family_rows = []

        # Fit every transform from the same frozen train window set.
        for index in tqdm(
            self.train_indices,
            desc="transforms",
            leave=False,
            unit="window",
        ):
            sample = self.dataset[index]
            mask = sample["mask"].cpu().numpy().astype(bool)
            if not mask.any():
                continue

            x_num_rows.append(sample["x_num"].cpu().numpy()[mask])
            x_cat_rows.append(sample["x_cat"].cpu().numpy()[mask])
            y_binary_rows.append(sample["y_binary"])
            y_family_rows.append(sample["y_family"])

        if not x_num_rows or not x_cat_rows:
            raise ValueError(
                "Training split did not contain any valid timesteps to fit transforms."
            )

        x_num_train = np.concatenate(x_num_rows, axis=0)
        x_cat_train = np.concatenate(x_cat_rows, axis=0)
        y_binary_train = np.asarray(y_binary_rows)
        y_family_train = np.asarray(y_family_rows)

        self.numeric_transform.fit(x_num_train)
        self.categorical_transform.fit(x_cat_train)
        self.target_transform.fit(y_binary_train, y_family_train)

        train_binary, train_family = cast(
            tuple[np.ndarray, np.ndarray],
            self.target_transform.transform(y_binary_train, y_family_train),
        )

        binary_encoder = self.target_transform.binary_encoder_
        family_encoder = self.target_transform.family_encoder_
        if binary_encoder is None or family_encoder is None:
            raise RuntimeError("TargetTransform must be fit before building train weights.")

        binary_weights = np.ones(len(binary_encoder.classes_), dtype=np.float32)
        family_weights = np.ones(len(family_encoder.classes_), dtype=np.float32)

        binary_classes = np.unique(train_binary).astype(np.int64)
        family_classes = np.unique(train_family).astype(np.int64)

        binary_weights[binary_classes] = compute_class_weight(
            class_weight="balanced",
            classes=binary_classes,
            y=train_binary,
        ).astype(np.float32)
        family_weights[family_classes] = compute_class_weight(
            class_weight="balanced",
            classes=family_classes,
            y=train_family,
        ).astype(np.float32)

        self.binary_loss_weight = torch.tensor(binary_weights, dtype=torch.float32)
        self.family_loss_weight = torch.tensor(family_weights, dtype=torch.float32)
        self.train_sample_weights = [
            float((binary_weights[int(y_binary)] * family_weights[int(y_family)]) ** 0.5)
            for y_binary, y_family in zip(train_binary, train_family, strict=False)
        ]

        return self

    def normalise(self, sample: dict[str, Any], train: bool = False) -> dict[str, torch.Tensor]:
        """Apply the fitted transforms to one raw sample."""
        x_num = sample["x_num"].cpu().numpy()
        x_cat = sample["x_cat"].cpu().numpy()
        mask = sample["mask"].cpu().numpy().astype(bool)

        x_num_out = np.zeros_like(x_num, dtype=np.float32)
        x_cat_out = np.zeros_like(x_cat, dtype=np.int64)

        if mask.any():
            x_num_valid = self.numeric_transform.transform(x_num[mask])
            x_cat_valid = self.categorical_transform.transform(x_cat[mask])

            if train and self.config.dataset.transforms.noise_std > 0.0:
                noise = np.random.normal(
                    loc=0.0,
                    scale=self.config.dataset.transforms.noise_std,
                    size=x_num_valid.shape,
                ).astype(np.float32)
                x_num_valid = x_num_valid + noise

            x_num_out[mask] = x_num_valid
            x_cat_out[mask] = x_cat_valid

        y_binary_out, y_family_out = cast(
            tuple[np.ndarray, np.ndarray],
            self.target_transform.transform(
                np.asarray([sample["y_binary"]], dtype=object),
                np.asarray([sample["y_family"]], dtype=object),
            ),
        )

        return {
            "x_num": torch.tensor(x_num_out, dtype=torch.float32),
            "x_cat": torch.tensor(x_cat_out, dtype=torch.long),
            "y_binary": torch.tensor(int(y_binary_out[0]), dtype=torch.long),
            "y_family": torch.tensor(int(y_family_out[0]), dtype=torch.long),
            "mask": sample["mask"],
        }

    def __len__(self) -> int:
        """Return the number of raw windows."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one transformed sample."""
        sample = cast(dict[str, Any], self.dataset[index])
        return self.normalise(sample, train=index < self._train_end)
