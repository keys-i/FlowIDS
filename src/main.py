"""
Main entrypoint for the NetFlow preprocessing and model pipeline.

This module runs the preprocessing workflow, train/validation/test
data, and launches training or evaluation routines.

Created By: Keys-i
ID: 49088276
"""

import torch

from src.config import load_config
from src.data import FlowDataset, build_loaders

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

config = load_config("config.toml")

# TODO: preprocess the dataset to build the dataset
dataset = FlowDataset(config)
train_loader, val_loader, test_loader, _ = build_loaders(dataset, config)

# TODO: initialise the model and device for GPU
# TODO: train the dataset
# TODO: visualise the dataset
# TODO: test the dataset
