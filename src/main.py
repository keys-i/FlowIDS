"""Train or evaluate M0 models."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.config import Config
from src.data.context import build
from src.data.dataset import CATEGORICAL, FlowDataset, collate
from src.data.features import NUMERIC_COLUMNS, PARTITION, SCORE
from src.data.load import load
from src.data.preprocess import PORT_BUCKET_START, State, transform
from src.data.preprocess import fit as fit_preprocess
from src.data.split import chronological, holdout
from src.eval import evaluate
from src.metrics import binary
from src.model.network import FlowTransformer
from src.obases import always_benign
from src.train import fit


def _device(name: str) -> torch.device:
    """Resolve the configured accelerator or choose the best local device."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed(value: int) -> None:
    """Set the local random seeds used by M0."""
    _ = random.seed(value)
    _ = torch.manual_seed(value)
    if torch.cuda.is_available():
        _ = torch.cuda.manual_seed_all(value)


def _sizes(state: State) -> list[int]:
    """Derive record-encoder vocabulary sizes from the fitted state."""
    sizes: list[int] = []
    for column in CATEGORICAL:
        if column.endswith("_range"):
            sizes.append(5)
        elif column.removesuffix("_id") in state["ports"]:
            values = state["ports"][column.removesuffix("_id")].values()
            sizes.append(max((PORT_BUCKET_START + 7, *values)) + 1)
        else:
            values = state["categorical"][column.removesuffix("_id")].values()
            sizes.append(max((2, *values)) + 1)
    return sizes


def _loader(
    datasets: list[FlowDataset],
    config: Config,
    *,
    shuffle: bool,
    device: torch.device,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Combine source partitions into one M0 loader."""
    if not datasets:
        raise ValueError("no scorable flows remain after splitting")
    return DataLoader(
        ConcatDataset(datasets),
        batch_size=config.data.batch_size,
        shuffle=shuffle,
        num_workers=config.data.workers,
        persistent_workers=config.data.workers > 0,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )


def _model(config: Config, state: State) -> FlowTransformer:
    """Build the configured model on the shared feature view."""
    sizes = _sizes(state)
    if config.model.kind == "base":
        return FlowTransformer(config, len(NUMERIC_COLUMNS), sizes, causal=False)
    if config.model.kind in {"small", "matched"}:
        return FlowTransformer(config, len(NUMERIC_COLUMNS), sizes, causal=True)
    raise ValueError("model.kind must be base, small, or matched")


def _partitions(
    frames: dict[str, pl.LazyFrame],
    state: State,
    config: Config,
    names: tuple[str, ...],
) -> dict[str, list[FlowDataset]]:
    """Transform each source and build contexts separately for every partition."""
    output: dict[str, list[FlowDataset]] = {name: [] for name in names}
    for frame in frames.values():
        events = transform(frame.filter(pl.col(PARTITION).is_in(names)), state).collect()
        for name in output:
            partition = events.filter(pl.col(PARTITION) == name)
            if partition.height and partition.filter(pl.col(SCORE)).height:
                contexts = build(partition, config.data.horizon_minutes, config.data.max_events)
                output[name].append(FlowDataset(partition, contexts, config.data.horizon_minutes))
    return output


def _development(config: Config) -> tuple[dict[str, pl.LazyFrame], pl.LazyFrame]:
    """Load and chronologically divide only the configured development sources."""
    frames = load(config)
    development = {
        source: chronological(
            frames[source],
            config.split.train_fraction,
            config.split.validation_fraction,
            config.split.purge_minutes,
        )
        for source in config.data.development
    }
    train = pl.concat(
        [frame.filter(pl.col(PARTITION) == "train") for frame in development.values()]
    )
    return development, train


def _save(path: Path, value: object) -> None:
    """Write a small JSON artifact with stable formatting."""
    _ = path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _contract(config: Config) -> dict[str, object]:
    """Return the small set of settings that must match at evaluation."""
    model: dict[str, object] = {"kind": config.model.kind}
    model.update(
        {
            "d_model": config.model.d_model,
            "layers": config.model.layers,
            "heads": config.model.heads,
            "ffn": config.model.ffn,
            "dropout": config.model.dropout,
        }
    )
    return {
        "sources": {"development": list(config.data.development), "holdout": config.data.holdout},
        "context": {
            "horizon_minutes": config.data.horizon_minutes,
            "max_events": config.data.max_events,
        },
        "split": {
            "train_fraction": config.split.train_fraction,
            "validation_fraction": config.split.validation_fraction,
            "purge_minutes": config.split.purge_minutes,
        },
        "model": model,
    }


def train(config: Config) -> None:
    """Train a model without including the configured holdout source."""
    _seed(config.run.seed)
    device = _device(config.run.device)
    output = Path(config.run.output)
    output.mkdir(parents=True, exist_ok=True)

    development, train_frame = _development(config)
    state = fit_preprocess(train_frame)
    partitions = _partitions(development, state, config, ("train", "validation"))
    model = _model(config, state).to(device)
    history, best_state = fit(
        model,
        _loader(partitions["train"], config, shuffle=True, device=device),
        _loader(partitions["validation"], config, shuffle=False, device=device),
        config,
        device,
    )
    torch.save(
        {
            "contract": _contract(config),
            "model": best_state,
            "preprocess": state,
        },
        output / "model.pt",
    )
    _save(output / "history.json", history)
    print(f"Saved {config.model.kind} to {output / 'model.pt'}")


def evaluate_model(config: Config) -> None:
    """Evaluate saved weights on development tests and the untouched holdout."""
    _seed(config.run.seed)
    device = _device(config.run.device)
    output = Path(config.run.output)
    bundle = torch.load(output / "model.pt", map_location=device, weights_only=True)
    state: State = bundle["preprocess"]
    if bundle["contract"] != _contract(config):
        raise ValueError("checkpoint contract does not match the configuration")
    model = _model(config, state).to(device)
    _ = model.load_state_dict(bundle["model"])

    development, _ = _development(config)
    results: dict[str, object] = {}
    for source, frame in development.items():
        source_partitions = _partitions({source: frame}, state, config, ("test",))
        _, labels, probability = evaluate(
            model,
            _loader(source_partitions["test"], config, shuffle=False, device=device),
            device,
        )
        results[source] = {
            config.model.kind: binary(labels, probability),
            "always_benign": binary(labels, always_benign(len(labels), labels.device)),
        }

    holdout_frame = holdout(load(config)[config.data.holdout])
    holdout_partitions = _partitions({config.data.holdout: holdout_frame}, state, config, ("test",))
    _, labels, probability = evaluate(
        model,
        _loader(holdout_partitions["test"], config, shuffle=False, device=device),
        device,
    )
    results[config.data.holdout] = {
        config.model.kind: binary(labels, probability),
        "always_benign": binary(labels, always_benign(len(labels), labels.device)),
    }
    _save(output / "metrics.json", results)
    print(json.dumps(results, indent=2, allow_nan=False))


def main() -> None:
    """Parse the model command line and run its selected action."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("action", choices=("train", "evaluate"))
    _ = parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config = Config.load(arguments.config)
    if arguments.action == "train":
        train(config)
    else:
        evaluate_model(config)


if __name__ == "__main__":
    main()
