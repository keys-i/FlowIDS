"""Minimal window-based flow dataset for PyTorch."""

from __future__ import annotations

from typing import Any

import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.config import Config

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .window_split import build_fixed_windows


class FlowDataset(Dataset):
    """Return one fixed temporal window as tensors."""

    def __init__(self, config: Config) -> None:
        targets_cfg = config.dataset.targets
        window_cfg = config.dataset.window
        engine = config.dataset.polars.engine

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
                .collect(engine=engine)
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
        """Return the number of windows."""
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one window as tensors."""
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
