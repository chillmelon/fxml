#!/usr/bin/env python
# coding: utf-8

"""
Comprehensive Forex Data Preprocessing Pipeline

This script consolidates feature engineering and preprocessing steps:
1. Data loading and validation
2. Feature engineering (returns, technical indicators, time features)
3. Normalization and scaling
4. Data export for ML training

Note: For data resampling, use the separate resampling.py script.

Usage:
    python preprocess_pipeline.py --config config/config.yaml --mode full
    python preprocess_pipeline.py --mode full --input data/resampled/USDJPY-58m-dollar-20210101-20241231.pkl --output data/processed/
    python preprocess_pipeline.py --mode features --input data/resampled/USDJPY-58m-dollar-20210101-20241231.pkl --output data/features.pkl
    python preprocess_pipeline.py -c config/config.yaml -m full -v  # Short options with verbose

Note: When using --config, the script will automatically:
- Assemble the resampled input filename from the 'resampling' section parameters
- Generate output filenames like: USDJPY-58m-dollar-20210101-20241231-processed.pkl
- Create files: USDJPY-58m-dollar-20210101-20241231-normalized.pkl, scalers, etc.
"""

import os
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands, DonchianChannel

from utils import build_file_paths_from_config
from utils import load_config as utils_load_config


def load_config(config_path):
    """Load configuration from YAML file - wrapper for utils function"""
    return utils_load_config(config_path)


def assemble_resampled_filename(config):
    """Assemble resampled filename from config parameters - uses utils function

    Args:
        config: Configuration dictionary with resampling section

    Returns:
        str: Full path to resampled file
    """
    paths, _, _ = build_file_paths_from_config(config)
    return str(paths["resampled"])


def assemble_output_filename(config, stage="processed"):
    """Assemble output filename from config parameters - uses utils function

    Args:
        config: Configuration dictionary with resampling section
        stage: Processing stage ('processed', 'normalized', 'features')

    Returns:
        str: Filename without directory path
    """
    paths, _, _ = build_file_paths_from_config(config)
    if stage in paths:
        return os.path.basename(str(paths[stage]))
    else:
        # Fallback for custom stages like 'features'
        processed_filename = os.path.basename(str(paths["processed"]))
        return processed_filename.replace("-processed.pkl", f"-{stage}.pkl")


def load_data(file_path):
    """Load data from pickle or CSV file"""
    click.echo(f"Loading data from {file_path}")

    if file_path.endswith(".pkl"):
        df = pd.read_pickle(file_path)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .pkl or .csv")

    click.echo(f"Data loaded with shape: {df.shape}")
    return df


def add_return_features(df, price_col="close", config=None):
    """Add return features (delta, return, log return)"""
    click.echo(f"Adding return features for {price_col}...")

    df = df.copy()
    df[f"{price_col}_delta"] = df[price_col] - df[price_col].shift(1)
    df[f"{price_col}_return"] = df[price_col] / df[price_col].shift(1) - 1
    df[f"{price_col}_log_return"] = np.log(df[price_col] / df[price_col].shift(1))

    # Get rolling mean windows from config or use defaults
    rolling_windows = [5, 10]
    if config and "preprocessing" in config and "features" in config["preprocessing"]:
        features_config = config["preprocessing"]["features"]
        if (
            "return_features" in features_config
            and "rolling_mean_windows" in features_config["return_features"]
        ):
            rolling_windows = features_config["return_features"]["rolling_mean_windows"]

    # Add rolling return means based on config
    for window in rolling_windows:
        df[f"ret_mean_{window}"] = (
            df[f"{price_col}_log_return"]
            .rolling(window=window, min_periods=window)
            .mean()
        )

    # Add log volume based on config
    add_log_volume = True
    if config and "preprocessing" in config and "features" in config["preprocessing"]:
        features_config = config["preprocessing"]["features"]
        if "return_features" in features_config:
            add_log_volume = features_config["return_features"].get("log_volume", True)

    if add_log_volume and "volume" in df.columns:
        df["log_volume"] = np.log1p(df["volume"])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def add_technical_indicators(df, config=None):
    """Add comprehensive technical indicators"""
    click.echo("Adding technical indicators...")

    df = df.copy()

    # Get technical indicator settings from config or use defaults
    ta_config = {}
    if config and "preprocessing" in config and "features" in config["preprocessing"]:
        features_config = config["preprocessing"]["features"]
        if "technical_indicators" in features_config:
            ta_config = features_config["technical_indicators"]

    # EMAs and slopes
    click.echo("→ EMAs and slopes")
    ema_windows = ta_config.get("ema_windows", [5, 20])
    for window in ema_windows:
        ema = EMAIndicator(close=df["close"], window=window)
        df[f"ema{window}"] = ema.ema_indicator()
        df[f"ema{window}_slope"] = df[f"ema{window}"].diff()

    # ATR
    click.echo("→ ATR")
    atr_windows = ta_config.get("atr_windows", [14, 20])
    for window in atr_windows:
        atr = AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=window
        )
        df[f"atr{window}"] = atr.average_true_range()

    # Volume-adjusted return and close-to-atr features (using first ATR window)
    atr_col = f"atr{atr_windows[0]}"
    if "close_log_return" in df.columns and atr_col in df.columns:
        df["vol_adj_return"] = df["close_log_return"] / df[atr_col]
    if "close_delta" in df.columns and atr_col in df.columns:
        df["close_to_atr"] = df["close_delta"] / df[atr_col]

    # ADX
    click.echo("→ ADX")
    atr_windows = ta_config.get("adx_windows", [14, 20])
    for window in atr_windows:
        adx = ADXIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=window,
            fillna=False,
        )
        df[f"adx{window}"] = adx.adx()
        df[f"plus_di{window}"] = adx.adx_pos()
        df[f"minus_di{window}"] = adx.adx_neg()

    # Bollinger Bands
    click.echo("→ Bollinger Bands")
    bb_config = ta_config.get("bollinger_bands", {"window": 20, "window_dev": 2})
    bb = BollingerBands(
        close=df["close"],
        window=bb_config["window"],
        window_dev=bb_config["window_dev"],
    )
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mavg"] = bb.bollinger_mavg()
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / df["bb_width"]

    # Donchian Channel
    click.echo("→ Donchian Channel")
    dc_config = ta_config.get("donchian_channel", {"window": 20})
    dc = DonchianChannel(
        high=df["high"], low=df["low"], close=df["close"], window=dc_config["window"]
    )
    df["donchian_upper"] = dc.donchian_channel_hband()
    df["donchian_lower"] = dc.donchian_channel_lband()
    df["donchian_mid"] = dc.donchian_channel_mband()
    df["donchian_width"] = df["donchian_upper"] - df["donchian_lower"]

    # Stochastic Oscillator
    click.echo("→ Stochastic Oscillator")
    stoch_config = ta_config.get("stochastic", {"window": 14, "smooth_window": 3})
    stoch = StochasticOscillator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=stoch_config["window"],
        smooth_window=stoch_config["smooth_window"],
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # RSI
    click.echo("→ RSI")
    rsi_config = ta_config.get("rsi", {"window": 14})
    rsi = RSIIndicator(close=df["close"], window=rsi_config["window"])
    df[f"rsi{rsi_config['window']}"] = rsi.rsi()

    # MACD
    click.echo("→ MACD")
    macd_config = ta_config.get(
        "macd", {"window_slow": 26, "window_fast": 12, "window_sign": 9}
    )
    macd = MACD(
        close=df["close"],
        window_slow=macd_config["window_slow"],
        window_fast=macd_config["window_fast"],
        window_sign=macd_config["window_sign"],
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    return df


def add_time_features(df, timestamp_col="timestamp"):
    """Add cyclical time features"""
    click.echo("🕒 Adding time features...")

    df = df.copy()

    # Ensure timestamp is datetime
    if df[timestamp_col].dtype != "datetime64[ns]":
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Unix time
    df["unix_time"] = df[timestamp_col].astype("int64") / 1e9

    # Extract components
    df["hour"] = df[timestamp_col].dt.hour
    df["dow"] = df[timestamp_col].dt.dayofweek  # Monday=0
    df["dom"] = df[timestamp_col].dt.day
    df["month"] = df[timestamp_col].dt.month

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["dom_sin"] = np.sin(2 * np.pi * df["dom"] / 31)
    df["dom_cos"] = np.cos(2 * np.pi * df["dom"] / 31)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def determine_scaler_columns_from_config(config):
    """Determine which columns need standard vs minmax scaler based on config"""
    
    # Base features that always get standard scaler
    base_features_std = ["close_log_return", "log_volume", "spread"]
    
    # Features that always get minmax scaler (bounded 0-100 or 0-1)
    base_features_minmax = []
    
    cols_to_std = base_features_std.copy()
    cols_to_minmax = base_features_minmax.copy()
    
    if not config or "preprocessing" not in config or "features" not in config["preprocessing"]:
        return cols_to_std, cols_to_minmax
        
    features_config = config["preprocessing"]["features"]
    
    # Add return feature columns based on config
    if features_config.get("add_returns", True):
        if "return_features" in features_config:
            return_config = features_config["return_features"]
            rolling_windows = return_config.get("rolling_mean_windows", [5, 10])
            for window in rolling_windows:
                cols_to_std.append(f"ret_mean_{window}")
    
    # Add technical indicator columns based on config
    if features_config.get("add_technical_indicators", True):
        ta_config = features_config.get("technical_indicators", {})
        
        # EMA features (standard scaler)
        ema_windows = ta_config.get("ema_windows", [5, 20])
        for window in ema_windows:
            cols_to_std.extend([f"ema{window}", f"ema{window}_slope"])
        
        # ATR features (standard scaler)  
        atr_windows = ta_config.get("atr_windows", [14, 20])
        for window in atr_windows:
            cols_to_std.append(f"atr{window}")
        
        # Volume-adjusted features (standard scaler)
        cols_to_std.extend(["vol_adj_return", "close_to_atr"])
        
        # ADX features (minmax scaler - bounded 0-100)
        adx_windows = ta_config.get("adx_windows", [14])
        for window in adx_windows:
            cols_to_minmax.extend([f"adx{window}", f"plus_di{window}", f"minus_di{window}"])
        
        # RSI features (minmax scaler - bounded 0-100)
        if "rsi" in ta_config:
            rsi_window = ta_config["rsi"].get("window", 14)
            cols_to_minmax.append(f"rsi{rsi_window}")
        
        # Bollinger Bands (standard scaler)
        if "bollinger_bands" in ta_config:
            cols_to_std.extend(["bb_upper", "bb_lower", "bb_mavg", "bb_width", "bb_position"])
            
        # Donchian Channel (standard scaler)
        if "donchian_channel" in ta_config:
            cols_to_std.extend(["donchian_upper", "donchian_lower", "donchian_mid", "donchian_width"])
        
        # Stochastic Oscillator (minmax scaler - bounded 0-100)
        if "stochastic" in ta_config:
            cols_to_minmax.extend(["stoch_k", "stoch_d"])
        
        # MACD (standard scaler)
        if "macd" in ta_config:
            cols_to_std.extend(["macd", "macd_signal", "macd_diff"])
    
    return cols_to_std, cols_to_minmax


def normalize_features(df, scaler_dir=None, file_prefix="", config=None):
    """Normalize features using StandardScaler and MinMaxScaler"""
    click.echo("🔧 Normalizing features...")

    df = df.copy()

    # Check if explicit normalization config exists, otherwise determine from TA config
    if (
        config
        and "preprocessing" in config
        and "normalization" in config["preprocessing"]
        and ("standard_scaler_features" in config["preprocessing"]["normalization"] 
             or "minmax_scaler_features" in config["preprocessing"]["normalization"])
    ):
        # Use explicit normalization config
        norm_config = config["preprocessing"]["normalization"]
        COLS_TO_STD = norm_config.get("standard_scaler_features", [])
        COLS_TO_MIN_MAX = norm_config.get("minmax_scaler_features", [])
        click.echo("Using explicit normalization configuration from config")
    else:
        # Determine scalers from technical indicators config
        COLS_TO_STD, COLS_TO_MIN_MAX = determine_scaler_columns_from_config(config)
        click.echo("Determined scaler columns from technical indicators configuration")

    # Filter columns that exist in dataframe
    COLS_TO_STD = [col for col in COLS_TO_STD if col in df.columns]
    COLS_TO_MIN_MAX = [col for col in COLS_TO_MIN_MAX if col in df.columns]

    # Initialize scalers
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    # Apply scaling
    if COLS_TO_STD:
        df[COLS_TO_STD] = standard_scaler.fit_transform(df[COLS_TO_STD])
        click.echo(f"StandardScaler applied to {len(COLS_TO_STD)} features")

    if COLS_TO_MIN_MAX:
        df[COLS_TO_MIN_MAX] = minmax_scaler.fit_transform(df[COLS_TO_MIN_MAX])
        click.echo(f"MinMaxScaler applied to {len(COLS_TO_MIN_MAX)} features")

    # Save scalers if directory provided
    if scaler_dir:
        os.makedirs(scaler_dir, exist_ok=True)
        if COLS_TO_STD:
            scaler_path = os.path.join(scaler_dir, f"{file_prefix}_standard_scaler.pkl")
            joblib.dump(standard_scaler, scaler_path)
        if COLS_TO_MIN_MAX:
            scaler_path = os.path.join(scaler_dir, f"{file_prefix}_minmax_scaler.pkl")
            joblib.dump(minmax_scaler, scaler_path)
        click.echo(f"Scalers saved to {scaler_dir}")

    return df, standard_scaler, minmax_scaler


def full_preprocessing_pipeline(input_path, output_dir, file_prefix=None, config=None):
    """Run complete preprocessing pipeline from resampled data to normalized features"""

    click.echo("=" * 60)
    click.echo("🚀 COMPREHENSIVE FOREX PREPROCESSING PIPELINE")
    click.echo("=" * 60)

    # Create output directories
    output_dir = Path(output_dir)
    processed_dir = output_dir / "processed"
    normalized_dir = output_dir / "normalized"
    scalers_dir = output_dir / "scalers"

    for dir_path in [processed_dir, normalized_dir, scalers_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(input_path)

    # Set timestamp as index if it exists and is not already the index
    if "timestamp" in df.columns and df.index.name != "timestamp":
        if df["timestamp"].dtype != "datetime64[ns]":
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")

    # Check config for feature flags
    skip_returns = False
    skip_tech_indicators = False
    skip_time_features = False

    if config and "preprocessing" in config and "features" in config["preprocessing"]:
        features_config = config["preprocessing"]["features"]
        skip_returns = not features_config.get("add_returns", True)
        skip_tech_indicators = not features_config.get("add_technical_indicators", True)
        skip_time_features = not features_config.get("add_time_features", True)

    # Add return features
    if not skip_returns:
        df = add_return_features(df, config=config)

    # Add technical indicators
    if not skip_tech_indicators:
        df = add_technical_indicators(df, config=config)

    # Add time features (reset index to bring timestamp back as column)
    df = df.reset_index()
    if not skip_time_features:
        df = add_time_features(df)

    # Drop NaN values
    click.echo(f"Dropping NaN values...")
    before_shape = df.shape
    df = df.dropna()
    click.echo(f"Shape before: {before_shape}, after: {df.shape}")

    # Generate output filename from config if not provided
    if config and (file_prefix is None or file_prefix == "processed"):
        try:
            processed_filename = assemble_output_filename(config, "processed")
            normalized_filename = assemble_output_filename(config, "normalized")
            scaler_prefix = assemble_output_filename(config, "scalers").replace(
                "-scalers.pkl", ""
            )
            click.echo(
                f"Using config-based filenames: {processed_filename}, {normalized_filename}"
            )
        except ValueError as e:
            click.echo(
                f"Warning: Could not assemble filenames from config ({e}), using defaults"
            )
            processed_filename = "processed_data.pkl"
            normalized_filename = "normalized_data.pkl"
            scaler_prefix = "scalers"
    else:
        processed_filename = f"{file_prefix}_processed.pkl"
        normalized_filename = f"{file_prefix}_normalized.pkl"
        scaler_prefix = file_prefix

    # Save processed data (with features, before normalization)
    processed_path = processed_dir / processed_filename
    df_processed = df.copy()
    if "timestamp" in df_processed.columns:
        df_processed = df_processed.set_index("timestamp")
    df_processed.to_pickle(processed_path)
    click.echo(f"Processed data saved to {processed_path}")

    # Normalize features
    df_normalized, std_scaler, mm_scaler = normalize_features(
        df, scalers_dir, scaler_prefix, config
    )

    # Save normalized data
    normalized_path = normalized_dir / normalized_filename
    if "timestamp" in df_normalized.columns:
        df_normalized = df_normalized.set_index("timestamp")
    df_normalized.to_pickle(normalized_path)
    click.echo(f"Normalized data saved to {normalized_path}")

    click.echo("=" * 60)
    click.echo("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
    click.echo(f"Final data shape: {df_normalized.shape}")
    click.echo(f"📂 Files saved in: {output_dir}")
    click.echo("=" * 60)

    return df_normalized


@click.command()
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Path to YAML config file"
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["full", "features"]),
    default="full",
    help="Processing mode: full pipeline or features only",
)
@click.option(
    "--input", "-i", "input_path", type=click.Path(exists=True), help="Input file path"
)
@click.option("--output", "-o", type=click.Path(), help="Output directory or file path")
@click.option("--prefix", "-p", default="processed", help="File prefix for outputs")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def main(config, mode, input_path, output, prefix, verbose):
    """Comprehensive Forex Data Preprocessing Pipeline

    This script handles feature engineering and preprocessing steps:
    1. Data loading and validation
    2. Feature engineering (returns, technical indicators, time features)
    3. Normalization and scaling
    4. Data export for ML training

    Examples:
        python preprocess_pipeline.py --config config/config.yaml --mode full
        python preprocess_pipeline.py --input data/resampled/USDJPY-58m-dollar-processed.pkl --mode features
    """

    if verbose:
        click.echo("Starting preprocessing pipeline in verbose mode...")

    # Load config if provided
    config_data = None
    if config:
        config_data = load_config(config)
        click.echo(f"Loaded config from {config}")

    # Determine input file
    if not input_path:
        if config_data:
            # Assemble resampled filename from config parameters
            try:
                input_path = assemble_resampled_filename(config_data)
                click.echo(f"Assembled resampled filename: {input_path}")
            except ValueError as e:
                # Fallback to raw data if resampling config is incomplete
                if "data" in config_data and "raw" in config_data["data"]:
                    input_path = config_data["data"]["raw"]
                    click.echo(
                        f"Warning: Could not assemble resampled filename ({e}). Using raw data: {input_path}"
                    )
                else:
                    raise click.ClickException(f"Cannot determine input file: {e}")
        else:
            raise click.ClickException(
                "Input file must be specified via --input or in config file"
            )

    # Check if input file exists
    if not os.path.exists(input_path):
        raise click.ClickException(f"Input file not found: {input_path}")

    if mode == "full":
        # Full preprocessing pipeline
        output_dir = output or "data"
        if verbose:
            click.echo(f"Running full pipeline: {input_path} -> {output_dir}")
        full_preprocessing_pipeline(input_path, output_dir, prefix, config_data)

    elif mode == "features":
        # Feature engineering only (assumes already resampled OHLCV data)
        if verbose:
            click.echo(f"Running features-only mode: {input_path}")

        df = load_data(input_path)

        # Ensure timestamp handling
        if "timestamp" in df.columns and df.index.name != "timestamp":
            if df["timestamp"].dtype != "datetime64[ns]":
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

        # Add all features based on config
        if not config_data or config_data["preprocessing"]["features"].get(
            "add_returns", True
        ):
            df = add_return_features(df, config=config_data)
        if not config_data or config_data["preprocessing"]["features"].get(
            "add_technical_indicators", True
        ):
            df = add_technical_indicators(df, config=config_data)
        df = df.reset_index()
        if not config_data or config_data["preprocessing"]["features"].get(
            "add_time_features", True
        ):
            df = add_time_features(df)
        df = df.dropna()

        # Save with features
        if output:
            output_path = output
        elif config_data:
            try:
                features_filename = assemble_output_filename(config_data, "features")
                output_path = f"data/processed/{features_filename}"
            except ValueError:
                output_path = input_path.replace(".pkl", "_features.pkl")
        else:
            output_path = input_path.replace(".pkl", "_features.pkl")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.to_pickle(output_path)
        click.echo(f"Features added and saved to {output_path}")

    if verbose:
        click.echo("✅ Preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main()
