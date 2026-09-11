"""Train and evaluate flow encoders on one NF3 dataset"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import polars as pl
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import collate, make_datasets, make_loader, vocabulary_sizes
from src.data.features import NUMERIC_COLUMNS, PARTITION
from src.data.load import load_split
from src.data.preprocess import State
from src.data.preprocess import fit as fit_preprocess
from src.m0.network import FlowTransformer
from src.m1.network import Pretrainer
from src.m2.data import HORIZONS, FutureDataset
from src.m2.network import M2Pretrainer
from src.metrics import binary
from src.pretrain import pretrain
from src.timing import Timing
from src.train import evaluate, fit


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


def run(
    config: Config,
    *,
    evaluate_only: bool = False,
    timing: Timing | None = None,
) -> None:
    """Prepare one dataset, train the chosen model, and write test results

    Args:
        config: Model, data, optimizer, and output settings
        evaluate_only: Load saved weights and preprocessing instead of training
        timing: Optional synchronized stage timer; pilots write to a separate output directory
    """
    started = perf_counter()
    mark: Callable[[str | None], None] = timing.mark if timing is not None else lambda stage: None
    mark("Setup")
    seed_all(config.run.seed)
    device = choose_device(config.run.device)
    output = Path(config.run.output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "status.json", {"complete": False})
    print(f"{config.model.kind} on {device}: no application time limit")
    print(f"Loading {config.data.dataset}")
    mark("Load/split")
    frame = load_split(config)
    mark("Preprocess")
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
    mark("Model/setup")
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
        pretraining = config.model.kind in {
            "reconstruct",
            "teacher",
            "hybrid",
            "future-hybrid",
            "future-jepa",
        }
        mark("Histories")
        partitions = make_datasets(
            frame, state, config, ("train", "validation"), supervised=not pretraining
        )
        if pretraining:
            print(f"Pretraining on {len(partitions['train']):,} targets")
            mark("Model/setup")
            objective = (
                M2Pretrainer(model, config.model.kind)
                if config.model.kind in {"hybrid", "future-hybrid", "future-jepa"}
                else Pretrainer(model, config.model.kind)
            ).to(device)
            loaders: dict[str, DataLoader[dict[str, torch.Tensor]]] = {}
            eligibility: dict[str, object] = {}
            future: FutureDataset | None = None
            for name, dataset in partitions.items():
                if config.model.kind.startswith("future-"):
                    mark("Future targets")
                future = (
                    FutureDataset(dataset, frame.filter(pl.col(PARTITION) == name))
                    if config.model.kind.startswith("future-")
                    else None
                )
                loaders[name] = make_loader(
                    future if future is not None else dataset,
                    config,
                    shuffle=name == "train",
                    device=device,
                    collate_fn=future.collate if future is not None else collate,
                )
                if future is not None:
                    eligibility[name] = {
                        "anchors": len(dataset),
                        "eligible": len(future),
                        "horizons": {
                            str(h): int((future.targets[:, i] >= 0).sum())
                            for i, h in enumerate(HORIZONS)
                        },
                    }
            if eligibility:
                save_json(output / "future_targets.json", eligibility)
                print(f"Future targets: {eligibility}")
            seed_all(config.run.seed)
            mark("Pretrain setup")
            history = pretrain(
                objective,
                loaders["train"],
                loaders["validation"],
                config,
                device,
                timing=timing,
            )
            mark("Checkpoints")
            torch.save(
                {**checkpoint, "pretrainer": objective.state_dict(), "model": model.state_dict()},
                output / "pretrained.pt",
            )
            save_json(output / "pretrain_history.json", history)
            mark("Histories")
            del objective, loaders, future
            partitions = {
                name: dataset.with_labels(frame.filter(pl.col(PARTITION) == name))
                for name, dataset in partitions.items()
            }
            seed_all(config.run.seed)
        print(
            f"Classification: {len(partitions['train']):,} training and",
            f"{len(partitions['validation']):,} validation targets",
        )
        mark("Classify setup")
        history, weights = fit(
            model,
            make_loader(partitions["train"], config, shuffle=True, device=device),
            make_loader(partitions["validation"], config, shuffle=False, device=device),
            config,
            device,
            timing=timing,
        )
        mark("Checkpoints")
        _ = model.load_state_dict(weights)
        torch.save({**checkpoint, "model": weights}, output / "model.pt")
        save_json(output / "history.json", history)
        save_json(output / "config.json", checkpoint["config"])
        print(f"Saved classifier to {output / 'model.pt'}")
        del partitions, weights

    mark("Histories")
    dataset = make_datasets(frame, state, config, ("test",))["test"]
    del frame, checkpoint
    print(f"Evaluating {len(dataset):,} test targets")
    mark("Test")
    _, labels, probability = evaluate(
        model,
        make_loader(dataset, config, shuffle=False, device=device),
        device,
    )
    mark("Metrics/write")
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
        {"complete": True, "elapsed_seconds": perf_counter() - started},
    )
    print(json.dumps(results, indent=2, allow_nan=False))
    print(f"Saved results to {output}")
    mark(None)


def main(argv: list[str] | None = None) -> None:
    """Run one model from config through training and evaluation"""
    args = list(sys.argv[1:] if argv is None else argv)
    max_mode = args[:1] == ["max"]
    if max_mode:
        _ = args.pop(0)
    parser = argparse.ArgumentParser(
        description="Train to the epoch limit or early stopping; 'max' selects a separate folder",
        usage="%(prog)s [max] {M0,M1,M2} variant [--seed SEED] [--evaluate-only]",
    )
    _ = parser.add_argument("model", choices=("M0", "M1", "M2"))
    _ = parser.add_argument("variant")
    _ = parser.add_argument("--seed", type=int)
    _ = parser.add_argument("--evaluate-only", action="store_true")
    arguments = parser.parse_args(args)
    variants = {
        "M0": ("base", "small", "matched"),
        "M1": ("reconstruct", "teacher"),
        "M2": ("hybrid", "future-hybrid", "future-jepa"),
    }
    if arguments.variant not in variants[arguments.model]:
        parser.error(f"{arguments.model} variants: {', '.join(variants[arguments.model])}")
    path = Path("tools/config") / f"{arguments.model.lower()}.{arguments.variant.lower()}.toml"
    config = Config.load(path)
    expected_output = f"results/{arguments.model}-{arguments.variant}"
    if config.model.kind != arguments.variant or config.run.output != expected_output:
        parser.error(f"{path} must use variant {arguments.variant} and output {expected_output}")
    if max_mode:
        config.run.output = str(Path(config.run.output) / "max")
    if arguments.seed is not None:
        config.run.seed = arguments.seed
        config.run.output = str(Path(config.run.output) / f"seed-{arguments.seed}")
    run(config, evaluate_only=arguments.evaluate_only)


if __name__ == "__main__":
    main()
