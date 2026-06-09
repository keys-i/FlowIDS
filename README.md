# FlowIDS

Research code for NetFlow pretraining and intrusion detection.

This repository currently converts NetFlow CSV files to Parquet and explores
them with DuckDB. Model code will start with M0 under `src/`.

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

Exploration is complete. M0 implementation is starting under `src/`; the model
launcher is ready, but training and evaluation are not implemented yet.
