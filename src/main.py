"""
Main entrypoint for the NetFlow preprocessing and model pipeline.

This module runs the preprocessing workflow, train/validation/test
data, and launches training or evaluation routines.

Created By: Keys-i
ID: 49088276
"""

from pathlib import Path

import torch

from src.config import load_config
from src.data import FlowDataset, build_loaders
from src.eval import evaluate, plot_confusion_matrix, plot_training_curves, save_summary
from src.model import FlowTransformer
from src.train import fit

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

config = load_config("config.toml")
dataset = FlowDataset(config)
train_loader, val_loader, test_loader, state = build_loaders(dataset, config)

categorical_transform = state["categorical_transform"]
feature_encoder = categorical_transform.feature_encoder_
family_encoder = categorical_transform.family_encoder_
if feature_encoder is None or family_encoder is None:
    raise RuntimeError("Categorical encoders must be fit before the model is initialised.")

categorical_cardinalities = {
    name: len(categories) + 2
    for name, categories in zip(
        dataset.categorical_features,
        feature_encoder.categories_,
        strict=True,
    )
}

device = torch.device(config.model.device)
if device.type == "cuda" and not torch.cuda.is_available():
    print(f"requested device: {config.model.device}")
    device = torch.device("cpu")

model_cfg = vars(config.model)
train_cfg = vars(config.train)

aux_class_dims = (
    {key: int(value) for key, value in vars(config.model.aux_class_dims).items()}
    if "aux_class_dims" in model_cfg
    else None
)
aux_reg_dims = (
    {key: int(value) for key, value in vars(config.model.aux_reg_dims).items()}
    if "aux_reg_dims" in model_cfg
    else None
)
aux_gaussian_dims = (
    {key: int(value) for key, value in vars(config.model.aux_gaussian_dims).items()}
    if "aux_gaussian_dims" in model_cfg
    else None
)

model = FlowTransformer(
    num_numeric=len(dataset.numeric_features),
    categorical_cardinalities=categorical_cardinalities,
    num_families=len(family_encoder.classes_),
    d_model=config.model.d_model,
    nhead=config.model.nhead,
    num_layers=config.model.num_layers,
    dim_feedforward=config.model.dim_feedforward,
    max_len=config.dataset.window.len,
    dropout=config.model.dropout,
    aux_class_dims=aux_class_dims,
    aux_reg_dims=aux_reg_dims,
    aux_gaussian_dims=aux_gaussian_dims,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.train.lr,
    weight_decay=config.train.weight_decay,
)

epochs = int(train_cfg["epochs"]) if "epochs" in train_cfg else 10
amp = bool(train_cfg["amp"]) if "amp" in train_cfg else device.type == "cuda"
grad_clip_norm = float(train_cfg["grad_clip_norm"]) if "grad_clip_norm" in train_cfg else None
loss_weights = (
    {name: float(value) for name, value in vars(config.train.loss_weights).items()}
    if "loss_weights" in train_cfg
    else None
)

scheduler = None
if "scheduler" in train_cfg:
    scheduler_cfg = vars(config.train.scheduler)
    scheduler_name = str(scheduler_cfg["name"]).lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(scheduler_cfg["t_max"])
        )
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_cfg["step_size"]),
            gamma=float(scheduler_cfg["gamma"]),
        )
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_cfg["factor"]),
            patience=int(scheduler_cfg["patience"]),
        )
    else:
        raise ValueError(f"Unsupported scheduler name: {scheduler_name}.")

out_dir = Path(config.paths.out)
checkpoint_dir = out_dir / "checkpoints"
plots_dir = out_dir / "plots"
summary_path = out_dir / "summary.json"
out_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

history, fit_summary = fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs,
    checkpoint_dir,
    device=device,
    amp=amp,
    grad_clip_norm=grad_clip_norm,
    loss_weights=loss_weights,
    binary_class_weight=state["binary_loss_weight"],
    family_class_weight=state["family_loss_weight"],
    scheduler=scheduler,
)

best_checkpoint = torch.load(fit_summary["best_checkpoint_path"], map_location=device)
model.load_state_dict(best_checkpoint["model_state_dict"])

evaluation = evaluate(
    model,
    test_loader,
    device,
    amp=amp,
    loss_weights=loss_weights,
    binary_class_weight=state["binary_loss_weight"],
    family_class_weight=state["family_loss_weight"],
    history=history,
)

plot_training_curves(history, plots_dir)

family_plot_path = plots_dir / "family_confusion_matrix.png"
plot_confusion_matrix(
    evaluation["family_targets"],
    evaluation["family_predictions"],
    family_plot_path,
    labels=sorted(set(evaluation["family_targets"]) | set(evaluation["family_predictions"])),
    title="Family Confusion Matrix",
)

binary_plot_path = plots_dir / "binary_confusion_matrix.png"
plot_confusion_matrix(
    evaluation["binary_targets"],
    evaluation["binary_predictions"],
    binary_plot_path,
    labels=[0, 1],
    display_labels=["benign", "malicious"],
    title="Binary Confusion Matrix",
)

summary = save_summary(
    summary_path,
    history,
    evaluation,
    {
        "best": fit_summary["best_checkpoint_path"],
        "final": fit_summary["final_checkpoint_path"],
    },
)

print(f"device: {device}")
print(f"train batches: {len(train_loader)}")
print(f"val batches: {len(val_loader)}")
print(f"test batches: {len(test_loader)}")
print(f"optimizer: {optimizer.__class__.__name__}")
print(f"epochs: {epochs}")
print(f"best epoch: {summary['best_epoch']}")
print(f"test total loss: {evaluation['total_loss']:.4f}")
print(f"test binary acc: {evaluation['binary_acc']:.4f}")
print(f"test family acc: {evaluation['family_acc']:.4f}")
print(f"summary: {summary_path}")
