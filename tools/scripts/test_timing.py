"""Check timing extrapolation, real model stages, and Slurm argument forwarding"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO
from itertools import count
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.data.features import END_TIME, NUMERIC_COLUMNS, RAW_COLUMNS, ROUTING_COLUMNS, START_TIME
from src.main import run


def main() -> None:
    """Check full-loop sampling, uncapped training, and scheduler argument forwarding"""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        capture = root / "arguments"
        sbatch = root / "sbatch"
        _ = sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$TIMING_TEST_ARGS"\n')
        sbatch.chmod(0o700)
        environment = {
            key: value for key, value in os.environ.items() if not key.startswith("SLURM_")
        }
        environment.update(PATH=f"{root}:{environment['PATH']}", TIMING_TEST_ARGS=str(capture))
        for script, mode in (
            ("slurm.sh", []),
            ("slurm.sh", ["max"]),
            ("slurm.sh", ["timing"]),
            ("slurm.sh", ["max", "timing"]),
            ("slurm.test.sh", []),
        ):
            result = subprocess.run(
                ["bash", f"tools/scripts/{script}", *mode, "--time=01:30:00", "--array=5-7"],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
            times = [arg for arg in capture.read_text().splitlines() if arg.startswith("--time=")]
            assert times[-1] == "--time=01:30:00", times
            if mode:
                assert capture.read_text().splitlines()[-len(mode) :] == mode

        from src.timing import Timing
        from tools.scripts.timing import array_indices, budget_status, duration_seconds
        from tools.scripts.timing import main as timing_main

        assert array_indices("5-7") == [5, 6, 7]
        assert array_indices("0,3-5,5") == [0, 3, 4, 5]
        assert duration_seconds("01:30:00") == 5400
        assert duration_seconds("3-00:00:00") == 259200
        for invalid in ("-1", "7-5", "8", "5;6", ""):
            try:
                _ = array_indices(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid array accepted: {invalid}")

        timer = Timing(torch.device("cpu"), sample_batches=3, warmup=1)
        timer.mark("Classify train")
        loader = DataLoader(TensorDataset(torch.ones(10, 1)), batch_size=1)
        with patch("src.timing.stamp", side_effect=[0, 100, 110, 120, 130]):
            assert len(list(timer.training(loader))) == 4
        assert timer.samples["Classify train"]["epoch_ns"] == 190
        assert timer.samples["Classify train"]["steady_epoch_ns"] == 100
        timer.elapsed["Classify train"] = 130
        timer.elapsed["Classify validation"] = 20
        estimates = timer.project(train_epochs=2, pretrain_epochs=0)
        assert estimates["Classify train"] == 290
        assert estimates["Classify validation"] == 40
        assert budget_status({"Classify train": 60_000_000_000}, 100) == "Fits"
        assert budget_status({"Classify train": 70_000_000_000}, 100) == "Fits"
        assert budget_status({"Test": 100_000_000_000}, 100) == "Fits"
        assert (
            budget_status({"Classify train": 70_000_000_000, "Test": 40_000_000_000}, 100)
            == "Exceeds allocation"
        )

        torch.set_num_threads(1)
        rows = 160
        generator = random.Random(42)
        frame = pl.DataFrame(
            {
                column: [generator.randrange(1, 1000) for _ in range(rows)]
                if column in NUMERIC_COLUMNS
                else [1] * rows
                for column in RAW_COLUMNS
            }
        ).with_columns(
            pl.Series(START_TIME, [i * 1000 for i in range(rows)]),
            pl.Series(END_TIME, [i * 1000 + 1 for i in range(rows)]),
            pl.Series("Label", [i % 2 for i in range(rows)]),
            pl.lit("A").alias(ROUTING_COLUMNS[0]),
            pl.lit("B").alias(ROUTING_COLUMNS[1]),
        )
        frame.write_parquet(root / "tiny.parquet")
        config: Config | None = None
        for path in sorted(Path("tools/config").glob("m[012].*.toml")):
            config = Config.load(path)
            config.data.root, config.data.dataset = str(root), "tiny"
            config.data.workers, config.data.batch_size, config.data.max_events = 0, 8, 4
            config.split.purge_minutes = 0
            config.model.d_model, config.model.heads, config.model.ffn = 16, 2, 32
            config.model.layers, config.model.dropout = 4, 0.0
            config.run.device, config.run.output = "cpu", str(root / path.stem)
            config.train.epochs = 2
            if hasattr(config, "pretrain"):
                config.pretrain.epochs = 2
            timer = Timing(torch.device("cpu"), sample_batches=2, warmup=1)
            with (
                patch.dict(os.environ, {"SLURM_JOB_END_TIME": "1"}),
                patch("src.main.perf_counter", side_effect=count(0, 1_000_000)),
                patch("src.train.perf_counter", side_effect=count(1_000_000_000, 1_000_000)),
                patch("src.pretrain.perf_counter", side_effect=count(1_000_000_000, 1_000_000)),
                redirect_stdout(StringIO()),
            ):
                run(config, timing=timer)
            timer.mark(None)
            assert "Classify train" in timer.samples
            assert {
                "Load/split",
                "Preprocess",
                "Histories",
                "Test",
                "Metrics/write",
            } <= timer.elapsed.keys()
            output = Path(config.run.output)
            assert len(json.loads((output / "history.json").read_text())) == 1
            assert json.loads((output / "status.json").read_text())["complete"] is True
            assert len(pl.read_parquet(output / "predictions.parquet")) > 8
            if config.model.kind.startswith("future-"):
                assert "Future targets" in timer.elapsed
            if config.model.kind in {"hybrid", "future-hybrid"}:
                assert timer.elapsed["Loss balancing"] > 0
            print(f"{path.stem}: pilot and full validation/test stages passed")

        assert config is not None
        output = root / "timings"
        command = [
            "timing",
            "--array",
            "5",
            "--device",
            "cpu",
            "--mode",
            "max",
            "--time",
            "01:30:00",
            "--output",
            str(output),
        ]
        for full in (False, True):
            arguments = command + (["--full"] if full else [])
            if full:
                arguments[arguments.index("--time") + 1] = "96:00:00"
            with (
                patch.object(Config, "load", return_value=config),
                patch.object(sys, "argv", arguments),
                patch.dict(os.environ, environment, clear=True),
                patch("src.main.perf_counter", side_effect=count(0, 1_000_000)),
                patch("src.train.perf_counter", side_effect=count(1_000_000_000, 1_000_000)),
                patch("src.pretrain.perf_counter", side_effect=count(1_000_000_000, 1_000_000)),
                redirect_stdout(StringIO()),
            ):
                timing_main()
            name = "full-max" if full else "pilot-max"
            result = json.loads((output / "M2-hybrid" / f"{name}.json").read_text())
            assert result["complete"] is True
            seconds = 345600 if full else 5400
            assert result["limits_seconds"] == {"normal": seconds, "max": seconds}
            if full:
                assert result["measured_ns"] == result["stages_ns"]
                assert (
                    len(json.loads((output / "M2-hybrid/full-max/history.json").read_text())) == 2
                )
            else:
                assert not list((output / "M2-hybrid").glob("pilot-*/model.pt"))
            table = StringIO()
            with patch.object(sys, "argv", arguments + ["--report"]), redirect_stdout(table):
                timing_main()
            assert ("96h 00m" if full else "1h 30m") in table.getvalue()
            assert "Total" in table.getvalue()

        config.data.dataset = "missing"
        failed_table = StringIO()
        with (
            patch.object(Config, "load", return_value=config),
            patch.object(sys, "argv", command),
            patch.dict(os.environ, environment, clear=True),
            redirect_stdout(failed_table),
        ):
            try:
                timing_main()
            except SystemExit as error:
                assert error.code == 1
            else:
                raise AssertionError("failed pilots must exit unsuccessfully")
        failed = json.loads((output / "M2-hybrid/pilot-max.json").read_text())
        assert failed["complete"] is False and "FileNotFoundError" in failed["error"]
        assert "Load/split" in failed_table.getvalue()
        assert "measured" in failed_table.getvalue() and "Incomplete" in failed_table.getvalue()
        print("Timing math, all eight model loops, and Slurm forwarding passed")


if __name__ == "__main__":
    main()
