"""Time synchronized stages and extrapolate a short, real training pilot"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice
from time import perf_counter_ns
from typing import final

import torch
from torch.utils.data import DataLoader


def stamp(device: torch.device) -> int:
    """Wait for outstanding device work before reading the monotonic nanosecond clock"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    return perf_counter_ns()


@final
class Timing:
    """Keep nonoverlapping stage times, with optional training-only sampling"""

    def __init__(self, device: torch.device, *, sample_batches: int = 0, warmup: int = 5) -> None:
        """Use zero sample batches to measure an ordinary, unsampled run"""
        if sample_batches < 0 or warmup < 0:
            raise ValueError("sample batches and warmup must be nonnegative")
        self.device = device
        self.sample_batches = sample_batches
        self.warmup = warmup
        self.samples: dict[str, dict[str, int | float]] = {}
        self.current: str | None = None
        started = perf_counter_ns()
        self.started = stamp(device)
        self.elapsed: dict[str, int] = {"Device init": self.started - started}

    def mark(self, stage: str | None) -> None:
        """Close the previous stage and start the next without double counting"""
        now = stamp(self.device)
        if self.current is not None:
            self.elapsed[self.current] = self.elapsed.get(self.current, 0) + now - self.started
        self.current, self.started = stage, now

    def training[T](self, loader: DataLoader[T]) -> Iterator[T]:
        """Sample shuffled real batches, including data loading and optimizer work

        Synchronize after warmup and at three block boundaries, preserving overlap
        within each block. Full validation and test loaders are never sampled
        """
        if not self.sample_batches:
            yield from loader
            return
        stage = self.current
        if stage is None or not len(loader):
            raise ValueError("training timing needs an active stage and a nonempty loader")
        total = min(len(loader), self.warmup + self.sample_batches)
        warmup = min(self.warmup, total - 1)
        block = max(1, (total - warmup + 2) // 3)
        started = boundary = stamp(self.device)
        cold = measured = block_count = 0
        rates: list[float] = []
        for index, batch in enumerate(islice(loader, total), 1):
            yield batch
            if index == warmup:
                boundary = stamp(self.device)
                cold = boundary - started
            elif index > warmup:
                block_count += 1
                if block_count == block or index == total:
                    now = stamp(self.device)
                    measured += now - boundary
                    rates.append((now - boundary) / block_count)
                    boundary, block_count = now, 0
        rate = measured / (total - warmup)
        self.samples[stage] = {
            "observed_ns": cold + measured,
            "epoch_ns": cold + round(rate * (len(loader) - warmup)),
            "steady_epoch_ns": round(rate * len(loader)),
            "spread_percent": (max(rates) - min(rates)) / rate * 100 if rate else 0,
        }

    def project(self, *, train_epochs: int, pretrain_epochs: int) -> dict[str, int]:
        """Project configured epochs, keeping startup and loss balancing as one-off costs"""
        projected = dict(self.elapsed)
        if not self.sample_batches:
            return projected
        for prefix, epochs in (("Classify", train_epochs), ("Pretrain", pretrain_epochs)):
            if not epochs:
                continue
            stage = f"{prefix} train"
            if stage not in self.samples:
                raise ValueError(f"pilot did not finish {stage}; no complete estimate available")
            sample = self.samples[stage]
            # ponytail: assume steady epoch cost; use --full to measure early stopping and drift
            projected[stage] = (
                self.elapsed[stage]
                - int(sample["observed_ns"])
                + int(sample["epoch_ns"])
                + int(sample["steady_epoch_ns"]) * (epochs - 1)
            )
            for name in (f"{prefix} validation", f"{prefix} select"):
                if name in projected:
                    projected[name] *= epochs
        return projected
