# Handoff

**Updated:** 11 September 2026 from the local M0, M1, and M2 implementations.
No real-data or cluster run is recorded.

Start with the [repository README](../README.md#run-a-model). The [model ladder](plan/Model.md) separates the implemented models from later
plans; [paper notes](notes/README.md) explain the research.

## What exists

- M0 Base, Small, and Matched under `src/m0/`
- One `NF-CSE-CIC-IDS2018-v3` chronological 70/15/15 split with a purge
- Training and evaluation together in `src/train.py`
- Data preparation, batching, and context construction in `src/data/dataset.py`
- `src/main.py:run` owns one path from loading to test predictions
- Loading and splitting are together in `src/data/load.py`; the file is read once
- M1 shares feature tensors between pretraining and classification
- M1 validation reuses student/teacher states for diagnostics
- Full checkpoints: model, training, preprocessing, and input configuration
- `pixi run model M0 <base|small|matched>` runs training then evaluation
- M1 reconstruct and M1 teacher under `src/m1/`, run with `pixi run model M1 reconstruct` or `M1 teacher`
- M2 hybrid, future-hybrid, and future-jepa use the same fine-tuning path
- Shared pretraining in `src/pretrain.py`; future selection in `src/m2/data.py`
- All variants use the same single NF3 dataset
- Saved per-flow predictions and `pixi run plot` for figures and error examples
- `--seed` writes separate runs; comparison error bars show sample standard deviation
- `model` uses a three-hour budget; `model max` allows 72 hours and writes under `max/`
- Static checks through `pixi run lint`; no test directory

M0 is a FlowTransformer-style baseline with local implementation changes. It
is not a reproduction claim or a new step in the research ladder. The inline
always-benign result remains; Logistic and the old `obases.py` module are gone.

The 256-event variants keep the target and up to 255 earlier flows. Batches no
longer carry unused elapsed-time, position, or causal tensors. Context building
uses native Polars and metric calculation is vectorized on CPU.

Each epoch records `train_seconds`, `validation_seconds`,
`train_flows_per_second`, and `peak_cuda_bytes`. The last value is `None` when
CUDA is off.

The data refactor matched the previous splits, score flags, M1 weights, and
predictions in CPU checks. Labelled views share feature tensors. Full dataset
and GPU performance are still unmeasured.

## What remains

- Real-data and Bunya runs
- Multi-seed reported metrics
- GPU throughput and memory profiling

The dataset belongs at `data/NF-CSE-CIC-IDS2018-v3.parquet`; the README includes
the Hugging Face download command. It has not been downloaded here, and no
Slurm job has been submitted. Submission needs the real cluster account.

M1/M2 have CPU verification only. M2 future-jepa means future-vector loss
only; future-hybrid also keeps reconstruction and same-flow teacher loss.
Few-label and cross-network comparisons remain planned; M3 and later are not implemented.

## Cluster run

The [README](../README.md#run-on-bunya) selects three or 72 hours for the
5 variants available on this branch. The `m2` branch runs all eight.
Jobs need the real Slurm account. Incomplete runs stay out of plots; H100
runtime remains unmeasured.

## Conventions

Use Pixi and `tools/config/`. Keep experiment facts in their owning plan file.
Every paper read or used needs a note in `notes/`.
