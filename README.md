# FlowIDS

Research code for NetFlow pretraining and intrusion detection.

This repository currently converts NetFlow CSV files to Parquet and explores
them with DuckDB. Model code will start with M0 under `src/`.

## Setup

```bash
pixi install
pixi run -e dev check
```

## Convert data

```bash
pixi run convert -- -i data/raw -o data/parquet
```

## Explore data

```bash
pixi run duckdb < tools/scripts/exploration/profile.sql
```

The other exploration queries are in `tools/scripts/exploration/`. Data stays
under the ignored `data/` directory. The research plan is in `plan/`.

D0 is not complete. There is no model, split, or preprocessing code yet.
