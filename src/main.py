"""
Main entrypoint for the NetFlow preprocessing and model pipeline.

This module runs the preprocessing workflow, train/validation/test
data, and launches training or evaluation routines.

Created By: Keys-i
ID: 49088276
"""

import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

DATA_DIR = "../data/v3/pqt"
NF_DATASET = "NF-CICIDS2018-v3.parquet"

# TODO: preprocess the dataset to build the dataset
CSV_PATH = f"{DATA_DIR}/{NF_DATASET}"
# build_dataset function usage here

# TODO: initialise the model and device for GPU

# TODO: train the dataset

# TODO: visualise the dataset

# TODO: test the dataset

# main CLI flag handler for everything to give it
