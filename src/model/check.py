"""Check the M0 model and input invariants."""

from __future__ import annotations

import tomllib
from pathlib import Path

import polars as pl
import torch
from torch import Tensor

from src.config import Config
from src.data.features import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from src.data.preprocess import fit, transform
from src.model.network import FlowTransformer
from src.model.record import RecordEncoder

CATEGORICAL_SIZES = (259, 1035, 1035, 259, 259, 259, 5, 5)


def _causal(length: int) -> Tensor:
    """Return the boolean future-event mask for one sequence length."""
    return torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)


def _inputs(length: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return deterministic valid event tensors for a single sequence."""
    numeric = torch.arange(length * len(NUMERIC_COLUMNS), dtype=torch.float32).reshape(
        1, length, len(NUMERIC_COLUMNS)
    )
    missing = torch.zeros_like(numeric, dtype=torch.bool)
    categorical = torch.ones(1, length, len(CATEGORICAL_SIZES), dtype=torch.int64)
    elapsed = torch.zeros(1, length, 2)
    return numeric, missing, categorical, elapsed


def _missing_fields() -> None:
    """Assert fully missing fields retain neutral values and missingness."""
    frame = pl.DataFrame(
        [
            *(pl.Series(column, [None, None], dtype=pl.Float64) for column in NUMERIC_COLUMNS),
            *(pl.Series(column, [None, None], dtype=pl.Int64) for column in CATEGORICAL_COLUMNS),
        ]
    ).lazy()
    state = fit(frame)
    output = transform(frame, state).collect()
    assert all(output[column].to_list() == [0.0, 0.0] for column in NUMERIC_COLUMNS)
    assert all(output[f"{column}_missing"].to_list() == [1, 1] for column in NUMERIC_COLUMNS)
    _ = RecordEncoder(1, (3,), 8)


def _model(path: str, causal: bool) -> FlowTransformer:
    """Construct one configured model in deterministic evaluation mode."""
    model = FlowTransformer(Config.load(path), len(NUMERIC_COLUMNS), CATEGORICAL_SIZES, causal)
    return model.eval()


def _logits(
    model: FlowTransformer,
    numeric: Tensor,
    missing: Tensor,
    categorical: Tensor,
    elapsed: Tensor,
    padding: Tensor,
) -> Tensor:
    """Run one model through the common batch interface."""
    length = numeric.shape[1]
    return model(
        numeric,
        missing,
        categorical,
        elapsed,
        padding,
        torch.arange(length).unsqueeze(0),
        _causal(length),
    )


def main() -> None:
    """Assert M0's parameter, causal, and left-padding contracts."""
    _missing_fields()
    _ = torch.manual_seed(0)
    _ = torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    base_path = Path("tools/config/m0.base.toml")
    matched_path = Path("tools/config/m0.matched.toml")
    small_path = Path("tools/config/m0.small.toml")
    with base_path.open("rb") as file:
        base_config = tomllib.load(file)
    with matched_path.open("rb") as file:
        matched_config = tomllib.load(file)
    assert base_config["data"] == matched_config["data"]
    assert base_config["split"] == matched_config["split"]
    assert base_config["train"] == matched_config["train"]
    assert {key: value for key, value in base_config["model"].items() if key != "kind"} == {
        key: value for key, value in matched_config["model"].items() if key != "kind"
    }
    assert {key: value for key, value in base_config["run"].items() if key != "output"} == {
        key: value for key, value in matched_config["run"].items() if key != "output"
    }

    base = _model(str(base_path), causal=False)
    matched = _model(str(matched_path), causal=True)
    base_parameters = sum(parameter.numel() for parameter in base.parameters())
    matched_parameters = sum(parameter.numel() for parameter in matched.parameters())
    assert base_parameters == matched_parameters
    assert 23_750_000 <= base_parameters <= 26_250_000

    baseline = _model(str(small_path), causal=False)
    m0 = _model(str(small_path), causal=True)
    assert m0.causal
    assert sum(parameter.numel() for parameter in m0.parameters()) < 1_000_000
    value = torch.randn(1, 4, 64)
    padding = torch.tensor([[True, True, False, False]])
    padded = m0.backbone(value, padding, True)
    assert torch.equal(padded[padding], torch.zeros_like(padded[padding]))

    padding = torch.zeros(1, 4, dtype=torch.bool)
    original = m0.backbone(value, padding, True)
    changed = value.clone()
    changed[:, -1] += 1
    altered = m0.backbone(changed, padding, True)
    baseline_original = baseline.backbone(value, padding, False)
    baseline_altered = baseline.backbone(changed, padding, False)
    assert torch.equal(original[:, :-1], altered[:, :-1])
    assert not torch.equal(baseline_original[:, :-1], baseline_altered[:, :-1])

    numeric, missing, categorical, elapsed = _inputs(3)
    padding = torch.zeros(1, 3, dtype=torch.bool)
    padded_numeric = torch.nn.functional.pad(numeric, (0, 0, 2, 0))
    padded_missing = torch.nn.functional.pad(missing, (0, 0, 2, 0))
    padded_categorical = torch.nn.functional.pad(categorical, (0, 0, 2, 0))
    padded_elapsed = torch.nn.functional.pad(elapsed, (0, 0, 2, 0))
    padded_padding = torch.tensor([[True, True, False, False, False]])
    for model in (baseline, m0):
        logit = _logits(model, numeric, missing, categorical, elapsed, padding)
        padded_logit = _logits(
            model,
            padded_numeric,
            padded_missing,
            padded_categorical,
            padded_elapsed,
            padded_padding,
        )
        assert torch.allclose(logit, padded_logit, atol=1e-6, rtol=0)
        assert torch.isfinite(logit).all() and torch.isfinite(padded_logit).all()

    assert all(
        torch.isfinite(tensor).all()
        for tensor in (padded, original, altered, baseline_original, baseline_altered)
    )


if __name__ == "__main__":
    main()
