"""Temporal split, train-only transforms"""

from __future__ import annotations
from typing import Optional

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, RobustScaler
from torch.utils.data import Dataset, Subset


class NumericTransform:
    """Clip and robust-scale numeric features."""

    def __init__(
        self,
        clip_quantiles: tuple[float, float] = (1.0, 99.0),
    ) -> None:
        self.clip_quantiles = clip_quantiles
        self.lower_: Optional[np.ndarray] = None
        self.upper_: Optional[np.ndarray] = None
        self.scaler_: Optional[RobustScaler] = None

    def fit(self, x: np.ndarray) -> "NumericTransform":
        """Fit clipping bounds and scaler on training data."""
        if x.ndim != 2:
            raise ValueError("x must have shape [n_samples, n_features].")

        q_low, q_high = self.clip_quantiles
        self.lower_ = np.percentile(x, q_low, axis=0)
        self.upper_ = np.percentile(x, q_high, axis=0)

        clipped = np.clip(x, self.lower_, self.upper_)
        self.scaler_ = RobustScaler()
        self.scaler_.fit(clipped)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Clip and scale numeric features."""
        if self.lower_ is None or self.upper_ is None or self.scaler_ is None:
            raise RuntimeError("NumericTransform must be fit before transform().")

        clipped = np.clip(x, self.lower_, self.upper_)
        return self.scaler_.transform(clipped).astype(np.float32)


class CategoricalTransform:
    """Encode categorical features and targets."""

    def __init__(self) -> None:
        self.feature_encoder_: Optional[OrdinalEncoder] = None
        self.binary_encoder_: Optional[LabelEncoder] = None
        self.family_encoder_: Optional[LabelEncoder] = None

    def fit(
        self,
        x_cat: np.ndarray,
        y_binary: np.ndarray,
        y_family: np.ndarray,
    ) -> "CategoricalTransform":
        """Fit feature and target encoders on training data."""
        if x_cat.ndim != 2:
            raise ValueError("x_cat must have shape [n_samples, n_features].")

        self.feature_encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        self.feature_encoder_.fit(x_cat)

        self.binary_encoder_ = LabelEncoder()
        self.binary_encoder_.fit(y_binary)

        self.family_encoder_ = LabelEncoder()
        self.family_encoder_.fit(y_family)

        return self

    def transform(self, x_cat: np.ndarray) -> np.ndarray:
        """Transform categorical input features."""
        if self.feature_encoder_ is None:
            raise RuntimeError("CategoricalTransform must be fit before transform().")

        return self.feature_encoder_.transform(x_cat).astype(np.int64) + 2

    def transform_targets(
        self,
        y_binary: np.ndarray,
        y_family: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform target labels."""
        if self.binary_encoder_ is None or self.family_encoder_ is None:
            raise RuntimeError(
                "CategoricalTransform must be fit before transform_targets()."
            )

        y_binary_out = self.binary_encoder_.transform(y_binary).astype(np.int64)
        y_family_out = self.family_encoder_.transform(y_family).astype(np.int64)

        return y_binary_out, y_family_out


class TransformedDataset(Dataset):
    """Subset a dataset and apply fit transforms."""

    def __init__(
        self,
        dataset: Dataset,
        indices: list[int],
        numeric_transform: NumericTransform,
        categorical_transform: CategoricalTransform,
        encoded_binary: np.ndarray,
        encoded_family: np.ndarray,
        train: bool = False,
        noise_std: float = 0.0,
    ) -> None:
        self.dataset = Subset(dataset, indices)
        self.numeric_transform = numeric_transform
        self.categorical_transform = categorical_transform
        self.encoded_binary = encoded_binary
        self.encoded_family = encoded_family
        self.train = train
        self.noise_std = noise_std

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one transformed sample."""
        sample = self.dataset[index]

        x_num = sample["x_num"].cpu().numpy()
        x_cat = sample["x_cat"].cpu().numpy()
        mask = sample["mask"].cpu().numpy().astype(bool)

        x_num_out = np.zeros_like(x_num, dtype=np.float32)
        x_cat_out = np.zeros_like(x_cat, dtype=np.int64)

        if mask.any():
            x_num_valid = self.numeric_transform.transform(x_num[mask])
            x_cat_valid = self.categorical_transform.transform(x_cat[mask])

            if self.train and self.noise_std > 0.0:
                noise = np.random.normal(
                    loc=0.0,
                    scale=self.noise_std,
                    size=x_num_valid.shape,
                ).astype(np.float32)
                x_num_valid = x_num_valid + noise

            x_num_out[mask] = x_num_valid
            x_cat_out[mask] = x_cat_valid

        return {
            "x_num": torch.tensor(x_num_out, dtype=torch.float32),
            "x_cat": torch.tensor(x_cat_out, dtype=torch.long),
            "y_binary": torch.tensor(self.encoded_binary[index], dtype=torch.long),
            "y_family": torch.tensor(self.encoded_family[index], dtype=torch.long),
            "mask": sample["mask"],
        }


def split(
    dataset: Dataset,
    splits: tuple[float, float, float] = (0.70, 0.15, 0.15),
    purge: int = 1,
) -> tuple[list[int], list[int], list[int]]:
    """Split windows chronologically with a purge gap."""
    if len(splits) != 3:
        raise ValueError("splits must contain exactly 3 values.")

    train_size, val_size, test_size = splits

    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("splits must sum to 1.0.")

    n = len(dataset)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train_indices = list(range(0, train_end))
    val_indices = list(range(min(train_end + purge, n), val_end))
    test_indices = list(range(min(val_end + purge, n), n))

    return train_indices, val_indices, test_indices
