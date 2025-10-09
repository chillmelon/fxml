import os
from pathlib import Path

import pandas as pd
import torch
import yaml


def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_data(file_path):
    """Load data from pickle or CSV file"""
    file_path = str(file_path)
    print(f"Loading data from {file_path}")

    if file_path.endswith(".pkl"):
        df = pd.read_pickle(file_path)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .pkl or .csv")

    print(f"Data loaded with shape: {df.shape}")
    return df


def get_device():
    """Get the best available device for PyTorch"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
