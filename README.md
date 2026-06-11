# FlowIDS

Research code for NetFlow pretraining and intrusion detection.

This repository converts NetFlow CSV files to Parquet, explores them with
DuckDB, and provides the supervised M0 runtime under `src/`.

## Setup

```bash
pixi install
pixi run lint
```

## Commands

```bash
pixi run fmt
pixi run lint
pixi run clean
pixi run model
```

`model` selects a TOML file from `tools/config/`, then runs `src/main.py` in
training or evaluation mode.

## Convert data

```bash
pixi run convert -- -i data/raw -o data/parquet
```

## Explore data

```bash
pixi run duckdb < tools/scripts/exploration/profile.sql
```

The other exploration queries are in `tools/scripts/exploration/`. Data stays
under the ignored `data/` directory. The research plan is in `docs/plan/`.

## M0

The launcher trains or evaluates three M0 models:

```bash
pixi run python -m src.main train --config tools/config/m0.base.toml
pixi run python -m src.main train --config tools/config/m0.small.toml
pixi run python -m src.main train --config tools/config/m0.matched.toml
```

Training writes `model.pt` and `history.json`; evaluation writes `metrics.json`
under the configured output directory. `m0.base.toml` is the 25M unrestricted
FlowTransformer-style model. `m0.small.toml` is the causal eight-flow model
with width 64, two layers, two heads, and FFN 128 for cheap screening.
`m0.matched.toml` is the 25M causal model matched to Base; its only
architectural difference is the attention mask. Use a different `run.output`
for each run.
