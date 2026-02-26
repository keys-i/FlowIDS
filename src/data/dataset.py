"""Minimal window-based flow dataset for PyTorch."""

from __future__ import annotations
from typing import Any

import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .window_split import build_fixed_windows

NUMERIC_FEATURES = [
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    "MIN_TTL",
    "MAX_TTL",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN",
    "TCP_WIN_MAX_OUT",
    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
]

CATEGORICAL_FEATURES = [
    "PROTOCOL",
    "L7_PROTO",
    "L4_SRC_PORT",
    "L4_DST_PORT",
]


class FlowDataset(Dataset):
    """Return one fixed temporal window as tensors."""

    def __init__(self, config: Any) -> None:
        self.numeric_features = NUMERIC_FEATURES
        self.categorical_features = CATEGORICAL_FEATURES
        self.window_len = config.dataset.window.len
        self.binary_target = config.dataset.targets.binary
        self.family_target = config.dataset.targets.family

        columns = (
            [config.dataset.time_col]
            + self.numeric_features
            + self.categorical_features
            + [self.binary_target, self.family_target]
        )

        self.df = (
            pl.scan_parquet(config.paths.data)
            .select(columns)
            .sort(config.dataset.time_col)
            .collect()
        )

        self.windows = build_fixed_windows(
            nrows=self.df.height,
            window_len=self.window_len,
            stride=config.dataset.window.stride,
            drop_incomplete=config.dataset.window.drop_incomplete,
            pad_value=config.dataset.window.pad_value,
        )

    def __len__(self) -> int:
        """Return the number of windows."""
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
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
            "y_binary": torch.tensor(
                target_row[self.binary_target],
                dtype=torch.float32,
            ),
            "y_family": torch.tensor(
                target_row[self.family_target],
                dtype=torch.long,
            ),
            "mask": torch.tensor(window_info["mask"], dtype=torch.bool),
        }
