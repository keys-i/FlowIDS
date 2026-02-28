"""
Window construction helpers for flow datasets
"""

from __future__ import annotations

import numpy as np


def build_fixed_windows(
    nrows: int,
    window_len: int,
    stride: int,
    drop_incomplete: bool = True,
    pad_value: int = -1,
):
    """Build deterministic fixed windows over row indices.

    Rules:
    - Preserves row order inside each window.
    - Stores mapping back to source row indices.
    - Uses exclusive `end` indexing.
    - If `drop_incomplete=True`, drops the final short tail window.
    - If `drop_incomplete=False`, pads the final short tail window with
      `pad_value` and emits a boolean mask.
    """
    if nrows < 0:
        raise ValueError("nrows must be non-negative")
    if window_len <= 0:
        raise ValueError("window_len must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    windows = []

    for start in range(0, nrows, stride):
        stop = start + window_len

        if stop <= nrows:
            row_indices = np.arange(start, stop, dtype=np.int64)
            mask = np.ones(window_len, dtype=bool)
            length = window_len
        else:
            length = nrows - start

            if length <= 0:
                break

            if drop_incomplete:
                break

            row_indices = np.full(window_len, pad_value, dtype=np.int64)
            row_indices[:length] = np.arange(start, nrows, dtype=np.int64)

            mask = np.zeros(window_len, dtype=bool)
            mask[:length] = True

        windows.append(
            {
                "row_indices": row_indices.tolist(),
                "mask": mask.tolist(),
                "start": int(start),
                "end": int(min(stop, nrows)),  # exclusive end
                "length": int(length),  # real, unpadded length
                "is_padded": bool(length < window_len),
            }
        )

    return windows
