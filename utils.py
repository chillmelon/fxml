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


def parse_threshold(threshold_str):
    """Parse threshold string like '123M', '58K', or '1000000' to float value

    Args:
        threshold_str: String or numeric threshold

    Returns:
        float: Threshold value
    """
    if isinstance(threshold_str, (int, float)):
        return float(threshold_str)

    if isinstance(threshold_str, str):
        threshold_str = threshold_str.upper().strip()

        if threshold_str.endswith("M"):
            return float(threshold_str[:-1]) * 1_000_000
        elif threshold_str.endswith("K"):
            return float(threshold_str[:-1]) * 1_000
        else:
            return float(threshold_str)

    raise ValueError(f"Invalid threshold format: {threshold_str}")


def convert_volume_millions_to_actual(
    df, ask_vol_col="askVolume", bid_vol_col="bidVolume", target_col="volume"
):
    """Convert volume from millions to actual volume units"""
    if ask_vol_col in df.columns and bid_vol_col in df.columns:
        df[target_col] = (df[ask_vol_col] + df[bid_vol_col]) * 1_000_000
    return df


def ensure_directory_exists(file_path):
    """Create directory for file path if it doesn't exist"""
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)


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
    resampling = config.get("resampling", {})
    symbol = resampling.get("symbol", "UNKNOWN").lower()
    date_range = resampling.get("date_range", {})
    start_date = normalize_date_format(date_range.get("start_date", "20200101"))
    end_date = normalize_date_format(date_range.get("end_date", "20241231"))

    # Build event name based on resampling type
    if resampling.get("type") == "dollar":
        threshold = resampling.get("threshold", 58000000)
        if threshold >= 1000000:
            threshold_str = f"{int(threshold/1000000)}m"
        else:
            threshold_str = str(int(threshold))
        sample_event = f"{threshold_str}-dollar"
    else:
        minutes = resampling.get("minutes", 5)
        sample_event = f"{minutes}m"

    # Build base names
    resampled_name = f"{symbol}-{sample_event}-{start_date}-{end_date}"
    label_event = "cusum_filter"  # Could be made configurable
    label_name = f"{resampled_name}-{label_event}"

    # Base directories
    base_path = Path(base_dir)

    # File paths
    paths = {
        "normalized": base_path / "normalized" / f"{resampled_name}-normalized.pkl",
        "direction_labels": base_path / "direction_labels" / f"{label_name}.pkl",
        "processed": base_path / "processed" / f"{resampled_name}-processed.pkl",
        "resampled": base_path / "resampled" / f"{resampled_name}.pkl",
        "labels": base_path / "labels" / f"{label_name}-labels.pkl",
        "meta_labels": base_path / "meta_labels" / f"{label_name}-meta.pkl",
    }

    return paths, resampled_name, sample_event, label_event
