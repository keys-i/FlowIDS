"""Estimate or measure each model's full pipeline and print stage and budget tables"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import traceback
from contextlib import redirect_stdout
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time
from typing import TypedDict, cast

import torch
from rich.console import Console
from rich.table import Table

from src.config import Config
from src.main import choose_device, run, save_json
from src.timing import Timing

VARIANTS = (
    ("M0", "base"),
    ("M0", "small"),
    ("M0", "matched"),
    ("M1", "reconstruct"),
    ("M1", "teacher"),
    ("M2", "hybrid"),
    ("M2", "future-hybrid"),
    ("M2", "future-jepa"),
)


class Report(TypedDict):
    """Store only timing results and the limits needed to redisplay them"""

    name: str
    complete: bool
    error: str | None
    sampled: bool
    stages_ns: dict[str, int]
    measured_ns: dict[str, int]
    spread_percent: float
    limits_seconds: dict[str, float]


def array_indices(value: str) -> list[int]:
    """Accept comma-separated indices and inclusive ranges from the model array"""
    indices: set[int] = set()
    for part in value.split(","):
        if not re.fullmatch(r"[0-7](?:-[0-7])?", part):
            raise ValueError("array must contain indices 0-7 or ranges such as 5-7")
        ends = [int(number) for number in part.split("-")]
        if ends[0] > ends[-1]:
            raise ValueError("array ranges must be ascending")
        indices.update(range(ends[0], ends[-1] + 1))
    return sorted(indices)


def duration_seconds(value: str) -> int:
    """Parse Slurm minutes, minutes:seconds, hours:minutes:seconds, or day forms"""
    if not re.fullmatch(r"(?:\d+-)?\d+(?::\d+){0,2}", value):
        raise ValueError("time must be positive minutes or [days-]HH:MM:SS")
    days, separator, clock = value.rpartition("-")
    parts = [int(part) for part in clock.split(":")]
    if any(part >= 60 for part in parts[1:]):
        raise ValueError("time minutes and seconds must be below 60")
    if separator:
        parts += [0] * (3 - len(parts))
    elif len(parts) == 1:
        parts.append(0)
    seconds = sum(part * 60**index for index, part in enumerate(reversed(parts)))
    seconds += int(days or "0") * 86400
    if seconds <= 0:
        raise ValueError("time must be positive")
    return seconds


def budget_status(stages: dict[str, int], seconds: float) -> str:
    """Compare the full pipeline with the scheduler allocation"""
    return "Fits" if sum(stages.values()) <= seconds * 1e9 else "Exceeds allocation"


def duration(value: int, *, nanoseconds: bool = False) -> str:
    """Keep the table readable unless raw nanoseconds were requested"""
    if nanoseconds:
        return f"{value:,}"
    seconds = value / 1e9
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = math.ceil(seconds / 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02}m" if hours else f"{minutes}m"


def show(reports: list[Report], *, nanoseconds: bool = False) -> None:
    """Display stage costs and compare normal/max scheduler allocations"""
    console = Console()
    for model, group in groupby(reports, key=lambda report: report["name"].split()[0]):
        variants = list(group)
        table = Table(title=f"{model} pipeline {'(ns)' if nanoseconds else ''}".rstrip())
        table.add_column("Stage")
        for report in variants:
            label = report["name"].split(maxsplit=1)[1]
            table.add_column(label if report["complete"] else f"{label}\nmeasured", justify="right")
        values = [
            report["stages_ns"] if report["complete"] else report["measured_ns"]
            for report in variants
        ]
        stages = list(dict.fromkeys(name for stages in values for name in stages))
        for stage in stages:
            table.add_row(
                stage,
                *(
                    duration(stages[stage], nanoseconds=nanoseconds) if stage in stages else "—"
                    for stages in values
                ),
            )
        table.add_row(
            "Total",
            *(
                duration(sum(report["stages_ns"].values()), nanoseconds=nanoseconds)
                if report["complete"]
                else "Incomplete"
                for report in variants
            ),
            style="bold",
        )
        if any(report["sampled"] for report in variants):
            table.add_row(
                "Train block spread",
                *(
                    f"{report['spread_percent']:.1f}%" if report["complete"] else "—"
                    for report in variants
                ),
            )
        console.print(table)
    budgets = Table("Model", "Normal allocation", "Normal", "Max allocation", "Max")
    for report in reports:
        cells = [report["name"]]
        for mode in ("normal", "max"):
            limit = report["limits_seconds"][mode]
            cells.extend(
                (
                    duration(round(limit * 1e9)),
                    budget_status(report["stages_ns"], limit)
                    if report["complete"]
                    else "Incomplete",
                )
            )
        budgets.add_row(*cells)
    console.print(budgets)
    if any(report["sampled"] for report in reports):
        console.print(
            "Totals estimate configured epochs; early stopping may finish sooner.",
            "Block spread is not an error bound.",
        )
    else:
        console.print("Measured runs include their actual early stopping.")
    for report in reports:
        if report["error"]:
            console.print(f"{report['name']}: {report['error']}", markup=False)


def main() -> None:
    """Run pilots on the allocated device, or redisplay their saved tables"""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--array", type=array_indices, default=os.environ.get("SLURM_ARRAY_TASK_ID", "0-7")
    )
    _ = parser.add_argument(
        "--batches", type=int, default=30, help="measured training batches per phase"
    )
    _ = parser.add_argument("--warmup", type=int, default=5, help="extra initial training batches")
    _ = parser.add_argument("--mode", choices=("normal", "max"), default="normal")
    _ = parser.add_argument(
        "--time", type=duration_seconds, help="scheduler allocation to compare, e.g. 72:00:00"
    )
    _ = parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    _ = parser.add_argument("--output", type=Path, default=Path("timings"))
    _ = parser.add_argument(
        "--report", action="store_true", help="read tables without running models"
    )
    _ = parser.add_argument(
        "--full", action="store_true", help="measure an ordinary run instead of sampling"
    )
    _ = parser.add_argument("--unit", choices=("human", "ns"), default="human")
    args = parser.parse_args()
    if args.batches < 1 or args.warmup < 0:
        parser.error("batches must be positive and warmup must be nonnegative")
    reports: list[Report] = []
    suffix = f"{'full' if args.full else 'pilot'}-{args.mode}"
    if args.report:
        for index in cast(list[int], args.array):
            model, variant = VARIANTS[index]
            path = args.output / f"{model}-{variant}" / f"{suffix}.json"
            if not path.is_file():
                parser.error(f"no timing report at {path}")
            reports.append(cast(Report, json.loads(path.read_text())))
        show(reports, nanoseconds=args.unit == "ns")
        if not all(report["complete"] for report in reports):
            parser.exit(1)
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable; run in the H100 allocation, or explicitly use --device cpu/mps"
        )
    device = choose_device(args.device)
    limit = args.time
    if end := os.environ.get("SLURM_JOB_END_TIME"):
        allocation = float(end) - float(os.environ.get("SLURM_JOB_START_TIME", time()))
        if not math.isfinite(allocation) or allocation <= 0:
            parser.error("Slurm allocation has no positive time limit remaining")
        limit = min(limit, allocation) if limit is not None else allocation
    limits = {
        mode: limit if limit is not None else hours * 3600
        for mode, hours in (("normal", 3), ("max", 72))
    }
    for index in cast(list[int], args.array):
        model, variant = VARIANTS[index]
        config = Config.load(f"tools/config/{model.lower()}.{variant}.toml")
        if config.model.kind != variant:
            parser.error(f"{model} {variant} configuration has a different model kind")
        config.run.device = args.device
        folder = args.output / f"{model}-{variant}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{suffix}.json"
        report: Report = {
            "name": f"{model} {variant}",
            "complete": False,
            "error": None,
            "sampled": not args.full,
            "stages_ns": {},
            "measured_ns": {},
            "spread_percent": 0,
            "limits_seconds": limits,
        }
        save_json(path, report)
        timer = Timing(device, sample_batches=0 if args.full else args.batches, warmup=args.warmup)
        print(
            f"Timing {model} {variant} on {device}; progress: {folder / (suffix + '.log')}",
            flush=True,
        )
        with (
            (folder / f"{suffix}.log").open("w") as log,
            TemporaryDirectory(dir=folder, prefix="pilot-") as temporary,
        ):
            config.run.output = str(folder / suffix) if args.full else temporary
            try:
                with redirect_stdout(log):
                    run(config, timing=timer)
                report["stages_ns"] = timer.project(
                    train_epochs=config.train.epochs,
                    pretrain_epochs=config.pretrain.epochs if hasattr(config, "pretrain") else 0,
                )
                report["complete"] = True
            except (ValueError, OSError, RuntimeError) as error:
                report["error"] = f"{type(error).__name__}: {error}"
                traceback.print_exc(file=log)
            finally:
                timer.mark(None)
                report["measured_ns"] = timer.elapsed
                report["spread_percent"] = max(
                    (sample["spread_percent"] for sample in timer.samples.values()), default=0
                )
                save_json(path, report)
        reports.append(report)
        show(reports, nanoseconds=args.unit == "ns")
    if not all(report["complete"] for report in reports):
        parser.exit(1)


if __name__ == "__main__":
    main()
