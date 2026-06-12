"""Train and evaluate flow encoders on one NF3 dataset"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from time import perf_counter, time

import polars as pl
import torch

from src.config import Config
from src.data.dataset import make_datasets, make_loader, vocabulary_sizes
from src.data.features import NUMERIC_COLUMNS, PARTITION
from src.data.load import load_split
from src.data.preprocess import State
from src.data.preprocess import fit as fit_preprocess
from src.m0.network import FlowTransformer
from src.metrics import binary
from src.train import TimeLimit, evaluate, fit


def choose_device(name: str) -> torch.device:
    """Resolve an explicit device, or try CUDA, MPS, then CPU for ``auto``"""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(value: int) -> None:
    """Seed Python and PyTorch without promising deterministic GPU kernels"""
    _ = random.seed(value)
    _ = torch.manual_seed(value)
    if torch.cuda.is_available():
        _ = torch.cuda.manual_seed_all(value)


def save_json(path: Path, value: object) -> None:
    """Write a JSON artifact, rejecting non-finite numbers"""
    _ = path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def input_settings(config: Config) -> dict[str, object]:
    """Record the input and architecture settings needed to reload weights"""
    return {
        "dataset": config.data.dataset,
        "context": {
            "horizon_minutes": config.data.horizon_minutes,
            "max_events": config.data.max_events,
        },
        "split": vars(config.split),
        "model": vars(config.model),
    }


def run(config: Config, *, evaluate_only: bool = False, hours: float = 3) -> None:
    """Prepare one dataset, train the chosen model, and write test results

    Args:
        config: Model, data, optimizer, and output settings
        evaluate_only: Load saved weights and preprocessing instead of training
        hours: Total budget, including data preparation and evaluation
    """
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError("hours must be finite and positive")
    started = perf_counter()
    seconds = hours * 3600
    if end := os.environ.get("SLURM_JOB_END_TIME"):
        seconds = min(seconds, float(end) - time())
    if seconds <= 0:
        raise TimeLimit("the Slurm allocation has already ended")
    deadline = started + seconds - min(60, seconds * 0.02)
    training_deadline = started + seconds * 0.8
    config.run.hours = hours
    seed_all(config.run.seed)
    device = choose_device(config.run.device)
    output = Path(config.run.output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "status.json", {"hours": hours, "complete": False})
    print(f"{config.model.kind} on {device}: {seconds / 3600:.2f} hours available")
    print(f"Loading {config.data.dataset}")
    frame = load_split(config)
    state: State
    if evaluate_only:
        checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
        if checkpoint["settings"] != input_settings(config):
            raise ValueError("checkpoint settings do not match the configuration")
        state = checkpoint["preprocess"]
    else:
        print("Fitting preprocessing on the training period")
        state = fit_preprocess(frame.filter(pl.col(PARTITION) == "train").lazy())
        checkpoint = {
            "settings": input_settings(config),
            "preprocess": state,
            "config": {
                name: vars(value)
                for name, value in vars(config).items()
                if isinstance(value, Config)
            },
        }
    model = FlowTransformer(
        config, len(NUMERIC_COLUMNS), vocabulary_sizes(state), causal=config.model.kind != "base"
    ).to(device)
    print(
        f"Model: {config.model.layers} layers, width {config.model.d_model},",
        f"{sum(parameter.numel() for parameter in model.parameters()):,} parameters",
    )

    if evaluate_only:
        _ = model.load_state_dict(checkpoint["model"])
    else:
        partitions = make_datasets(frame, state, config, ("train", "validation"))
        print(
            f"Classification: {len(partitions['train']):,} training and",
            f"{len(partitions['validation']):,} validation targets",
        )
        history, weights = fit(
            model,
            make_loader(partitions["train"], config, shuffle=True, device=device),
            make_loader(partitions["validation"], config, shuffle=False, device=device),
            config,
            device,
            deadline=training_deadline,
        )
        _ = model.load_state_dict(weights)
        torch.save({**checkpoint, "model": weights}, output / "model.pt")
        save_json(output / "history.json", history)
        save_json(output / "config.json", checkpoint["config"])
        print(f"Saved classifier to {output / 'model.pt'}")
        del partitions, weights

    dataset = make_datasets(frame, state, config, ("test",))["test"]
    del frame, checkpoint
    print(f"Evaluating {len(dataset):,} test targets")
    _, labels, probability = evaluate(
        model,
        make_loader(dataset, config, shuffle=False, device=device),
        device,
        deadline=deadline,
    )
    dataset.targets.with_columns(
        pl.Series("label", labels.numpy()),
        pl.Series("probability", probability.numpy()),
    ).write_parquet(output / "predictions.parquet")
    results = {
        config.data.dataset: {
            config.model.kind: binary(labels, probability),
            "always_benign": binary(labels, torch.zeros_like(labels)),
        }
    }
    save_json(output / "metrics.json", results)
    save_json(
        output / "status.json",
        {"hours": hours, "complete": True, "elapsed_seconds": perf_counter() - started},
    )
    print(json.dumps(results, indent=2, allow_nan=False))
    print(f"Saved results to {output}")


def main(argv: list[str] | None = None) -> None:
    """Run one model from config through training and evaluation"""
    args = list(sys.argv[1:] if argv is None else argv)
    hours = 72 if args[:1] == ["max"] else 3
    if hours == 72:
        _ = args.pop(0)
    parser = argparse.ArgumentParser(
        description="Train for up to 3 hours; prefix with 'max' for up to 72 hours",
        usage="%(prog)s [max] {M0} variant [--seed SEED] [--evaluate-only]",
    )
    _ = parser.add_argument("model", choices=("M0",))
    _ = parser.add_argument("variant")
    _ = parser.add_argument("--seed", type=int)
    _ = parser.add_argument("--evaluate-only", action="store_true")
    arguments = parser.parse_args(args)
    variants = {
        "M0": ("base", "small", "matched"),
    }
    if arguments.variant not in variants[arguments.model]:
        parser.error(f"{arguments.model} variants: {', '.join(variants[arguments.model])}")
    path = Path("tools/config") / f"{arguments.model.lower()}.{arguments.variant.lower()}.toml"
    config = Config.load(path)
    expected_output = f"results/{arguments.model}-{arguments.variant}"
    if config.model.kind != arguments.variant or config.run.output != expected_output:
        parser.error(f"{path} must use variant {arguments.variant} and output {expected_output}")
    if hours == 72:
        config.run.output = str(Path(config.run.output) / "max")
    if arguments.seed is not None:
        config.run.seed = arguments.seed
        config.run.output = str(Path(config.run.output) / f"seed-{arguments.seed}")
    try:
        run(config, evaluate_only=arguments.evaluate_only, hours=hours)
    except TimeLimit as error:
        parser.exit(1, f"Time limit: {error}. Run is incomplete; see {config.run.output}\n")


if __name__ == "__main__":
    main()
