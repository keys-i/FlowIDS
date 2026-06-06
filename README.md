# FlowIDS

Research code for NetFlow pretraining and intrusion detection.

This repository currently converts NetFlow CSV files to Parquet and explores
them with DuckDB. Model code will start with M0 under `src/`.

## Model runs

Model code lives on `m0`, `m1`, and `m2`; `m2` includes all eight variants.
From a model branch, use one of these commands:

```bash
pixi run model M0 small       # Three-hour budget
pixi run model max M0 small   # 72-hour budget
```

For the cluster, use `bash tools/scripts/slurm.sh --account=a_yourgroup`, or
add `max` immediately after `slurm.sh` for 72 hours. Existing epoch limits and
early stopping still apply. Max results use `results/<model>-<variant>/max/`.
Incomplete evaluations stay out of plots. H100 runtime has not been measured.

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
