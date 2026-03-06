"""
Main entrypoint for the NetFlow preprocessing and model pipeline

This file does four steps:
1. Load config and build the datasets/dataloaders
2. Build the model, optimizer, and scheduler
3. Train the model and reload the best checkpoint
4. Run final evaluation and save plots + summary

Created By: Keys-i
ID: 49088276
"""

from pathlib import Path

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

from src.config import load_config
from src.data import FlowDataset, TransformedDataset, build_loaders
from src.eval import evaluate, plot_confusion_matrix, plot_training_curves, save_summary
from src.model import FlowTransformer
from src.train import fit

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

to_dict = lambda typ, section: {key: typ(value) for key, value in vars(section).items()}

def build_scheduler(optimizer: Optimizer, scheduler_cfg):
    """Create the configured learning-rate scheduler"""
    name = str(scheduler_cfg.name).lower()

    match name:
        case "cosine":
            return CosineAnnealingLR(optimizer, T_max=int(scheduler_cfg.t_max))
        case "step":
            return StepLR(
                optimizer,
                step_size=int(scheduler_cfg.step_size),
                gamma=float(scheduler_cfg.gamma),
            )
        case "plateau":
            return ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(scheduler_cfg.factor),
                patience=int(scheduler_cfg.patience),
            )
        case _:
            raise ValueError(f"Unsupported scheduler name: {name}.")

# Load config
config = load_config("config.toml")

# == Build and transform dataset ==
dataset = FlowDataset(config)
transformed_dataset = TransformedDataset(dataset, config)

# DataLoader only handles batching, workers, and train sampling
train_dl, val_dl, test_dl = build_loaders(transformed_dataset, config)

# -- Choosing the device --
device = torch.device(config.model.device)

# == build the model ==
model_cfg, train_cfg = config.model, config.train
model = FlowTransformer(
    num_numeric=len(dataset.numeric_features),
    categorical_cardinalities=transformed_dataset.categorical_cardinalities,
    num_families=len(transformed_dataset.target_transform.family_encoder_.classes_),
    d_model=model_cfg.d_model,
    nhead=model_cfg.nhead,
    num_layers=model_cfg.num_layers,
    dim_feedforward=model_cfg.dim_feedforward,
    max_len=config.dataset.window.len,
    dropout=model_cfg.dropout,
    aux_class_dims=to_dict(int, model_cfg.aux_class_dims),
    aux_reg_dims=to_dict(int, model_cfg.aux_reg_dims),
    aux_gaussian_dims=to_dict(int, model_cfg.aux_gaussian_dims),
).to(device)

# == build the optimizer and scheduler ==
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_cfg.lr,
    weight_decay=train_cfg.weight_decay,
)
scheduler = build_scheduler(optimizer, train_cfg.scheduler)

# == train the model ==
# -- train settings --
epochs = int(train_cfg.epochs)
amp = bool(train_cfg.amp)
grad_clip_norm = float(train_cfg.grad_clip_norm)
loss_weights = to_dict(float, train_cfg.loss_weights)

# -- output paths --
out_dir = Path(config.paths.out)
checkpoint_dir, plots_dir = out_dir / "checkpoints", out_dir / "plots"
summary_path = out_dir / "summary.json"

for path in (out_dir, checkpoint_dir, plots_dir):
    path.mkdir(parents=True, exist_ok=True)

# =*- train -*=
history, fit_summary = fit(
    model,
    train_dl,
    val_dl,
    optimizer,
    epochs,
    checkpoint_dir,
    device=device,
    amp=amp,
    grad_clip_norm=grad_clip_norm,
    loss_weights=loss_weights,
    binary_class_weight=transformed_dataset.binary_loss_weight,
    family_class_weight=transformed_dataset.family_loss_weight,
    scheduler=scheduler,
)

# reload the best checkpoint for final evaluation
best_checkpoint = torch.load(fit_summary["best_checkpoint_path"], map_location=device)
model.load_state_dict(best_checkpoint["model_state_dict"])


# =*- evaluate -*=
evaluation = evaluate(
    model,
    test_dl,
    device,
    amp=amp,
    loss_weights=loss_weights,
    binary_class_weight=transformed_dataset.binary_loss_weight,
    family_class_weight=transformed_dataset.family_loss_weight,
    history=history,
)

# -- save plots --
plot_training_curves(history, plots_dir)

plot_confusion_matrix(
    evaluation["family_targets"],
    evaluation["family_predictions"],
    plots_dir / "family_confusion_matrix.png",
    labels=sorted(set(evaluation["family_targets"]) | set(evaluation["family_predictions"])),
    title="Family Confusion Matrix",
)

plot_confusion_matrix(
    evaluation["binary_targets"],
    evaluation["binary_predictions"],
    plots_dir / "binary_confusion_matrix.png",
    labels=[0, 1],
    display_labels=["benign", "malicious"],
    title="Binary Confusion Matrix",
)

# -- save summary --
summary = save_summary(
    summary_path,
    history,
    evaluation,
    {
        "best": fit_summary["best_checkpoint_path"],
        "final": fit_summary["final_checkpoint_path"],
    },
)

# -- print summary --
print(
    f""" === Summary ===
device: {device}
train batches: {len(train_dl)}
val batches: {len(val_dl)}
test batches: {len(test_dl)}
optimizer: {optimizer.__class__.__name__}
epochs: {epochs}
best epoch: {summary['best_epoch']}
test total loss: {evaluation['total_loss']:.4f}
test binary acc: {evaluation['binary_acc']:.4f}
test family acc: {evaluation['family_acc']:.4f}
summary: {summary_path}"""
)