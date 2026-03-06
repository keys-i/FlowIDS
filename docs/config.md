# Config
This defines each section that is used in `config.toml` and it's meaning.

## Paths
Defines where the pipeline reads the dataset from and where it writes all outputs.

| Key | Type | Example | Description |
|---|---|---:|---|
| `data` | string | `"data/NF-CICIDS2018-v3.parquet"` | Path to the input parquet dataset used for the pipeline. |
| `out` | string | `"out"` | Output directory where checkpoints, plots, and summaries are saved. |

## Dataset
Defines the core dataset-level settings used before model training starts.

| Key | Type | Example | Description |
|---|---|---:|---|
| `time_col` | string | `"FLOW_START_MILLISECONDS"` | Timestamp column used to sort flows before building temporal windows. |

### Targets
Defines which dataset columns are treated as the main supervised targets.

| Key | Type | Example | Description |
|---|---|---:|---|
| `binary` | string | `"Label"` | Column used for the main binary target, typically benign vs malicious. |
| `family` | string | `"Attack"` | Column used for the main multiclass attack-family target. |

### Windows
Defines how raw rows are grouped into fixed temporal windows for the Transformer.

| Key | Type | Example | Description |
|---|---|---:|---|
| `len` | integer | `8` | Number of timesteps per fixed window. |
| `stride` | integer | `8` | Step size between consecutive windows. If equal to `len`, windows do not overlap. |
| `drop_incomplete` | boolean | `false` | Whether to drop the final short window if it has fewer than `len` rows. |
| `pad_value` | integer | `-1` | Sentinel row-index value used before masking when padded windows are kept. |

### Transforms
Defines preprocessing applied to the dataset before it is fed into the model.

| Key | Type | Example | Description |
|---|---|---:|---|
| `clip_quantiles` | list[float] | `[1.0, 99.0]` | Lower and upper quantiles used to clip extreme numeric values before scaling. |
| `noise_std` | float | `0.0` | Standard deviation of Gaussian noise added to numeric features during training. `0.0` disables it. |

### Split
Defines how the dataset is divided into training, validation, and test sets.

| Key | Type | Example | Description |
|---|---|---:|---|
| `splits` | list[float] | `[0.7, 0.15, 0.15]` | Train, validation, and test fractions. These should sum to `1.0`. |
| `mode` | string | `"temporal"` | Split strategy. Here it means chronological splitting instead of random splitting. |
| `purge` | integer | `1` | Number of windows skipped between splits to reduce temporal leakage. |

### Polars
Defines settings for the Polars data-processing backend.

| Key | Type | Example | Description |
|---|---|---:|---|
| `engine` | string | `"cpu"` | Polars execution engine to use when reading and processing the dataset. |

## Dataloaders
Defines how batches are built and loaded during training and evaluation.

| Key | Type | Example | Description |
|---|---|---:|---|
| `batch_size` | integer | `256` | Number of windows per batch. |
| `num_workers` | integer | `4` | Number of DataLoader worker processes. |
| `pin_memory` | boolean | `true` | Whether to pin host memory for faster transfer to CUDA devices. |
| `balance_train` | boolean | `false` | Whether to use balanced sampling for the training loader. |
| `drop_last_train` | boolean | `false` | Whether to drop the last incomplete training batch. |
| `seed` | integer | `42` | Random seed used for reproducible loader behavior and sampling. |

## Model
Defines the main Transformer architecture and runtime device settings.

| Key | Type | Example | Description |
|---|---|---:|---|
| `device` | string | `"cuda"` | Device on which the model runs, for example `cpu` or `cuda`. |
| `d_model` | integer | `256` | Transformer hidden size used across embeddings and encoder layers. |
| `nhead` | integer | `8` | Number of attention heads. This should divide `d_model` exactly. |
| `num_layers` | integer | `6` | Number of Transformer encoder layers. |
| `dim_feedforward` | integer | `1024` | Hidden width of the feedforward sublayer inside each encoder block. |
| `dropout` | float | `0.1` | Dropout probability used in the model. |

### Aux Class Dims
Defines the output sizes of optional auxiliary classification heads.

| Key | Type | Example | Description |
|---|---|---:|---|
| `protocol` | integer | `256` | Output size for the auxiliary protocol classification head. |
| `l7_proto` | integer | `256` | Output size for the auxiliary L7 protocol classification head. |
| `src_port` | integer | `65538` | Output size for the auxiliary source-port classification head. Includes reserved ids used by encoding. |
| `dst_port` | integer | `65538` | Output size for the auxiliary destination-port classification head. Includes reserved ids used by encoding. |

### Aux Reg Dims
Defines the output sizes of optional auxiliary numeric regression heads.

| Key | Type | Example | Description |
|---|---|---:|---|
| `bytes` | integer | `2` | Output dimension of the auxiliary numeric regression head for byte targets, typically `IN_BYTES` and `OUT_BYTES`. |
| `pkts` | integer | `2` | Output dimension of the auxiliary numeric regression head for packet-count targets, typically `IN_PKTS` and `OUT_PKTS`. |

### Aux Gaussian Dims
Defines optional heteroscedastic Gaussian regression heads if you want the model to predict both means and variances.

| Key | Type | Example | Description |
|---|---|---:|---|
| *(empty table)* | table | — | Reserved for optional heteroscedastic Gaussian regression heads. Leave empty if unused. |

## Training
Defines the core optimizer and training loop settings.

| Key | Type | Example | Description |
|---|---|---:|---|
| `epochs` | integer | `10` | Number of full training passes over the training set. |
| `lr` | float | `1e-4` | Learning rate for the optimizer. |
| `weight_decay` | float | `1e-2` | Weight decay used by AdamW regularization. |
| `amp` | boolean | `true` | Whether to use automatic mixed precision during training and evaluation. |
| `grad_clip_norm` | float | `1.0` | Maximum gradient norm used for clipping. |

### Loss Weights
Defines how much each task contributes to the total multitask loss.

| Key | Type | Example | Description |
|---|---|---:|---|
| `binary` | float | `1.0` | Loss weight for the main binary classification task. |
| `family` | float | `1.0` | Loss weight for the main family classification task. |
| `protocol` | float | `0.2` | Loss weight for the auxiliary protocol classification task. |
| `l7_proto` | float | `0.2` | Loss weight for the auxiliary L7 protocol classification task. |
| `src_port` | float | `0.1` | Loss weight for the auxiliary source-port classification task. |
| `dst_port` | float | `0.1` | Loss weight for the auxiliary destination-port classification task. |
| `bytes` | float | `0.2` | Loss weight for the auxiliary numeric byte regression task. |
| `pkts` | float | `0.2` | Loss weight for the auxiliary numeric packet regression task. |

### Scheduler
Defines the learning-rate schedule used during training.

| Key | Type | Example | Description |
|---|---|---:|---|
| `name` | string | `"cosine"` | Learning-rate scheduler type. |
| `t_max` | integer | `10` | Main period parameter for cosine annealing. In this setup it usually matches the number of epochs. |

> [!NOTE] General Notes  
> These are a few important constraints and reminders that help keep the config valid and the experiments fair.
>
> | Topic | Note |
> |---|---|
> | Split fractions | `splits` should add up to `1.0`. |
> | Attention heads | `nhead` should divide `d_model` exactly. |
> | Auxiliary heads | If auxiliary heads are enabled in the model, their output dimensions and loss weights should be defined consistently. |
> | Temporal evaluation | `mode = "temporal"` and `purge = 1` are there to reduce leakage between train, validation, and test windows. |
> | Padding | `drop_incomplete = false` means incomplete final windows are kept and masked rather than discarded. |