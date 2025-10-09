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


def normalize_date_format(date_str):
    """Normalize date string to YYYYMMDD format

    Args:
        date_str: Date string in various formats (YYYY-MM-DD, YYYYMMDD, etc.)

    Returns:
        str: Date in YYYYMMDD format
    """
    if not date_str:
        return date_str

    # Remove common separators and whitespace
    date_str = str(date_str).replace("-", "").replace("/", "").replace(" ", "").strip()

    # If it looks like a datetime with time component, take just the date part
    if len(date_str) > 8:
        date_str = date_str[:8]

    # Validate it's 8 digits
    if len(date_str) == 8 and date_str.isdigit():
        return date_str

    # Try to parse with pandas and convert to YYYYMMDD
    try:
        parsed_date = pd.to_datetime(date_str)
        return parsed_date.strftime("%Y%m%d")
    except:
        raise ValueError(
            f"Cannot parse date: {date_str}. Expected format: YYYYMMDD or YYYY-MM-DD"
        )


def build_file_paths_from_config(config, base_dir="./data"):
    """Build standardized file paths from config parameters"""
    # Extract resampling config
    data_config = config.get("data", {})
    symbol = data_config.get("symbol", "UNKNOWN")
    date_range = data_config.get("date_range", {})
    start_date = date_range.get("start_date", "20200101")
    end_date = date_range.get("end_date", "20241231")

    # Build event name based on resampling type
    minutes = data_config.get("minutes", 5)

    # Build base names
    resampled_name = f"{symbol}-{minutes}m-{start_date}-{end_date}"
    event_name = config.get("data", {}).get("event_name")
    label_name = config.get("labeling", {}).get("type", "TB")

    # Base directories
    base_path = Path(base_dir)

    # File paths
    paths = {
        "normalized": base_path / "normalized" / f"{resampled_name}-normalized.pkl",
        "direction_labels": base_path
        / "direction_labels"
        / f"{event_name}-{label_name}.pkl",
        "processed": base_path / "processed" / f"{resampled_name}-processed.pkl",
        "resampled": base_path / "resampled" / f"{resampled_name}.pkl",
        "events": base_path / "events" / f"{event_name}.pkl",
        "labels": base_path
        / "labels"
        / f"{resampled_name}-{event_name}-{label_name}.pkl",
        "meta_labels": base_path / "meta_labels" / f"{event_name}-meta.pkl",
    }

    return paths, sample_event, label_event
