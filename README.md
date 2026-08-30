# FlowIDS

NetFlow intrusion detection with a supervised baseline (M0) and two pretraining
methods (M1/M2). Current runs use only `NF-CSE-CIC-IDS2018-v3`.

## Run a model

Model code lives on `m0`, `m1`, and `m2`. The `m0` branch has three M0
variants, `m1` has five M0/M1 variants, and `m2` has all eight.

From the project root, download the single dataset from
[Hugging Face](https://huggingface.co/datasets/keys-i/netFlow):

```bash
mkdir -p data
curl -fL --retry 3 --continue-at - \
  -o data/NF-CSE-CIC-IDS2018-v3.parquet \
  'https://huggingface.co/datasets/keys-i/netFlow/resolve/main/data/NF-CICIDS2018-v3.parquet'
```

The mirror uses the shorter filename `NF-CICIDS2018-v3.parquet` for this
dataset. All eight model configs read `data/NF-CSE-CIC-IDS2018-v3.parquet`.
After the download finishes:

```bash
pixi install
pixi run model M0 small
```

| Command | What it trains |
| --- | --- |
| `pixi run model M0 small` | Eight-flow causal baseline for checking the code |
| `pixi run model M0 base` | 25M baseline with attention across the supplied history |
| `pixi run model M0 matched` | Same shape as Base, with causal attention |
| `pixi run model M1 reconstruct` | Masked field reconstruction, then attack classification |
| `pixi run model M1 teacher` | Prediction of an unmasked teacher's vectors, then classification |
| `pixi run model M2 hybrid` | Reconstruction and same-flow teacher prediction together |
| `pixi run model M2 future-hybrid` | Hybrid loss plus later endpoint-related teacher targets |
| `pixi run model M2 future-jepa` | Future teacher-vector prediction only |

`model` has a three-hour budget. Add `max` before the model name for 72 hours:

```bash
pixi run model max M0 small
```

Both stop earlier if their epoch limit or early stopping is reached. Data
preparation counts toward the budget. Training reserves the final fifth for
test evaluation; M1/M2 split the training time between pretraining and
classification. Each phase also leaves time for validation. Batch counts and
timings are saved in the histories; three-hour runs may stop partway through an epoch.

Settings live in `tools/config/m0.*.toml`, `m1.*.toml`, and `m2.*.toml`. Pretrained variants
share a four-layer, 256-wide encoder. The teacher is a moving average of the
student and receives no gradient updates.

Every run uses a chronological 70/15/15 split with a purge around boundaries.
Preprocessing learns only from the training period. The run prints its device,
model size, target counts, and epoch losses. Results go to `results/<model>-<variant>/`.
The 72-hour runs use its `max/` subfolder. Use `--evaluate-only` with the same
command to score its saved checkpoint without training.

Time checks happen between batches. Slow data preparation, batches, or file
writes can overrun the local budget; Slurm enforces the allocation limit.
Weights are saved before test evaluation. A run whose full evaluation does
not finish stays marked incomplete in `status.json` and is excluded from plots.
If no classification validation finishes, the latest trained weights are saved.
Pretraining must finish at least one noncollapsed validation before fine-tuning.

## Follow the code

Start at `run()` in `src/main.py`. Read it from top to bottom:

1. `load_split` reads the file once and assigns the three time periods
2. `fit_preprocess` learns numeric scaling and category IDs from training flows
3. `make_datasets` builds bounded histories; `make_loader` pads them into batches
4. M1/M2 run `pretrain`, then every model runs classification with `fit`
5. The saved weights score the test period and save predictions

M1/M2 reuse the same feature tensors for pretraining and classification.
`with_labels` selects scorable targets without rebuilding histories.

The model components stay separate:

```text
src/m0/record.py    numeric values + missing flags + category IDs → flow vectors
src/m0/backbone.py  flow vectors + padding → contextual flow vectors
src/m0/heads.py     final flow vector → attack logit
src/m0/network.py   connects those three components

src/m1/masking.py  selects field groups to hide
src/m1/network.py  student, raw decoder, or EMA teacher and predictor
src/m2/network.py  hybrid losses and the future-only JEPA predictor
src/m2/data.py     later endpoint-related targets, kept outside student input
src/pretrain.py     shared pretraining, teacher updates, and collapse checks
src/train.py       classification loss, optimizer, and validation
```

| Problem | Where to stop in the debugger |
| --- | --- |
| Wrong split or missing data | `load_split` in `src/data/load.py` |
| Wrong history or padding | `FlowDataset.__getitem__` and `collate` in `src/data/dataset.py` |
| Wrong prediction shape | `FlowTransformer.forward` in `src/m0/network.py` |
| M1 loss or teacher problem | `Pretrainer.forward` in `src/m1/network.py` |
| M2 loss or future targets | `M2Pretrainer.losses` and `FutureDataset.collate` in `src/m2/` |
| Loss stops improving | The batch loop in `fit` or `pretrain` |

For tiny forward/backward checks without a dataset, run
`pixi run python -m src.m1.network` or `pixi run python -m src.m2.network`.

M2 future targets are the 1st, 4th, and 16th later flows touching either
endpoint, within the same split. Equal completion times are excluded. Teacher
histories use the same time/length limits; future records never enter student
inputs. Future variants pretrain on eligible anchors and fine-tune on all
scorable training targets.

Hybrid weights come from 200 training-only forward batches, using inverse
median losses scaled to sum to two. They stay fixed afterwards. `future-jepa`
has one loss with weight one. Check `pretrain_history.json` for weights and
warm-up cost, and `future_targets.json` for eligibility counts.

## Plot results

```bash
pixi run plot
```

This writes `results/figures/report.pdf`, PNG/SVG figures, a summary CSV, and
up to 100 mistaken predictions with flow IDs. Figures cover model comparisons,
learning curves, precision–recall, ROC, calibration, and confusion counts.

For training variation, run distinct seeds with identical settings:

```bash
pixi run model M1 teacher --seed 41
pixi run model M1 teacher --seed 42
pixi run model M1 teacher --seed 43
pixi run plot results/M1-teacher --output results/figures/M1-teacher
```

Each seed has its own folder. Error bars show one sample standard deviation
across seeds on the same split; they are not confidence intervals. One run has
no error bar. Error plots use threshold 0.5; pass `--threshold` only for a value
chosen on validation data. Older runs without `predictions.parquet` need
`--evaluate-only` before plotting errors.

## Run on Bunya

From the project root, with Pixi on `PATH` and the dataset available:

```bash
CONDA_OVERRIDE_CUDA=12.9 pixi install --frozen
bash tools/scripts/slurm.sh --account=a_yourgroup
```

For 72-hour jobs, submit this instead:

```bash
bash tools/scripts/slurm.sh max --account=a_yourgroup
```

Replace `a_yourgroup` with your account. The array runs one job per variant:
0 = M0 Base, 1 = Small, 2 = Matched, 3 = M1 reconstruct, 4 = M1 teacher,
5 = M2 hybrid, 6 = M2 future-hybrid, 7 = M2 future-jepa. Each requests one
H100, eight CPUs, 128 GB RAM, and three hours (72 with `max`). Install once before submission;
jobs use the existing environment. Logs go beside that model's results.

The array has no concurrency cap: all eight may run together if GPUs and account
limits allow. Do not add `%2` to `--array` if you want all eight running at once.
Eight GPUs for three hours use up to 24 GPU-hours; 72 hours allows 576 GPU-hours.
Queue time is separate. The script sets the [Slurm time limit](https://slurm.schedmd.com/sbatch.html#OPT_time)
when it submits the array. Full-data H100 runtime has not been measured.

## Other commands

```bash
pixi run fmt
pixi run lint
pixi run convert -- -i data/raw -o data/parquet
pixi run duckdb < tools/scripts/exploration/profile.sql
```

[Model ladder](docs/plan/Model.md) · [Research notes](docs/notes/README.md) ·
[Current status](docs/context.md)
