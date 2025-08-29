#!/usr/bin/env python
# coding: utf-8

"""
Forex Data Preprocessing Pipeline

Processing steps:
1. Feature engineering (returns, technical indicators, time features)
2. Normalization and scaling
3. Data export for ML training

Usage:
    python preprocessing.py --config config/config.yaml --mode full
    python preprocessing.py --input data/file.pkl --output data/processed/
"""

import os
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands, DonchianChannel

from utils import build_file_paths_from_config, load_config


def get_feature_config(config):
    """Helper to safely get feature configuration"""
    if (
        not config
        or "preprocessing" not in config
        or "features" not in config["preprocessing"]
    ):
        return {}
    return config["preprocessing"]["features"]


def handle_timestamp_index(df, timestamp_col="timestamp"):
    """Standardize timestamp handling - ensure timestamp is index"""
    if timestamp_col in df.columns and df.index.name != timestamp_col:
        if df[timestamp_col].dtype != "datetime64[ns]":
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit="ms")
        df = df.set_index(timestamp_col)
    return df


def handle_timestamp_column(df, timestamp_col="timestamp"):
    """Ensure timestamp is column for time feature extraction"""
    if df.index.name == timestamp_col:
        df = df.reset_index()
    return df


def load_data(file_path):
    """Load data from pickle or CSV file"""
    if str(file_path).endswith(".pkl"):
        return pd.read_pickle(file_path)
    elif str(file_path).endswith(".csv"):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .pkl or .csv")


def add_return_features(df, price_col="close", config=None):
    """Add return features (delta, return, log return)"""
    df = df.copy()
    df[f"{price_col}_delta"] = df[price_col] - df[price_col].shift(1)
    df[f"{price_col}_return"] = df[price_col] / df[price_col].shift(1) - 1
    df[f"{price_col}_log_return"] = np.log(df[price_col] / df[price_col].shift(1))

    features_config = get_feature_config(config)
    return_config = features_config.get("return_features", {})
    rolling_windows = return_config.get("rolling_mean_windows", [5, 10])

    # Add rolling return means
    for window in rolling_windows:
        df[f"ret_mean_{window}"] = (
            df[f"{price_col}_log_return"]
            .rolling(window=window, min_periods=window)
            .mean()
        )

    # Add log volume
    if return_config.get("log_volume", True) and "volume" in df.columns:
        df["log_volume"] = np.log1p(df["volume"])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def add_technical_indicators(df, config=None):
    """Add technical indicators"""
    df = df.copy()
    features_config = get_feature_config(config)
    ta_config = features_config.get("technical_indicators", {})

    # RV (Realized Volatility)
    rv_windows = ta_config.get("rv_windows", [5, 10, 15, 20])
    for window in rv_windows:
        df[f"rv{window}"] = df["close_log_return"].pow(2).rolling(window).sum()
        df[f"log_rv{window}"] = np.log1p(df[f"rv{window}"])
        df[f"sqrt_rv{window}"] = df[f"rv{window}"].pow(0.5)

    # EMAs
    ema_windows = ta_config.get("ema_windows", [5, 20])
    for window in ema_windows:
        ema = EMAIndicator(close=df["close"], window=window)
        df[f"ema{window}"] = ema.ema_indicator()
        df[f"ema{window}_slope"] = df[f"ema{window}"].diff()
        df[f"close_above_ema{window}"] = (df["close"] > df[f"ema{window}"]).astype(int)

    # ATR
    atr_windows = ta_config.get("atr_windows", [14, 20])
    for window in atr_windows:
        atr = AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=window
        )
        df[f"atr{window}"] = atr.average_true_range()
    # Cleanup 0.0 values from ATR calculations
    max_atr_column = f"atr{max(atr_windows)}"
    df[max_atr_column] = df[max_atr_column].replace(0.0, np.nan)
    df.dropna(inplace=True)
    for window in atr_windows:
        df[f"log_atr{window}"] = np.log1p(df[f"atr{window}"])
        df[f"atr{window}_percent"] = df[f"atr{window}"] / df["close"]
        df[f"atr{window}_adjusted_return"] = df["close_delta"] / df[f"atr{window}"]

    # ADX
    adx_windows = ta_config.get("adx_windows", [14, 20])
    for window in adx_windows:
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
    dc_windows = ta_config.get("donchian_channel_windows", [20])
    for window in dc_windows:
        dc = DonchianChannel(
            high=df["high"], low=df["low"], close=df["close"], window=window
        )
        df[f"dc{window}_upper"] = dc.donchian_channel_hband().shift(1)
        df[f"dc{window}_lower"] = dc.donchian_channel_lband().shift(1)
        df[f"dc{window}_mid"] = dc.donchian_channel_mband().shift(1)
        df[f"dc{window}_width"] = df[f"dc{window}_upper"] - df[f"dc{window}_lower"]
        df[f"close_above_dc{window}_mid"] = (
            df["close"] > df[f"dc{window}_mid"]
        ).astype(int)
        df[f"dc{window}_breakout"] = (df["close"] > df[f"dc{window}_upper"]).astype(int)
        df[f"dc{window}_breakdown"] = (df["close"] < df[f"dc{window}_lower"]).astype(
            int
        )

    # Stochastic Oscillator
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
    rsi_windows = ta_config.get("rsi_windows", [14, 20])
    for window in rsi_windows:
        rsi = RSIIndicator(close=df["close"], window=window)
        df[f"rsi{window}"] = rsi.rsi()
        df[f"rsi{window}_slope"] = df[f"rsi{window}"].diff()

    # MACD
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
    df = df.copy()

    # Ensure timestamp is datetime
    if df[timestamp_col].dtype != "datetime64[ns]":
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Unix time
    df["unix_time"] = df[timestamp_col].astype("int64") / 1e9

    # Extract components
    df["hour"] = df[timestamp_col].dt.hour
    df["dow"] = df[timestamp_col].dt.dayofweek
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


def get_scaler_columns(config):
    """Determine which columns need which scaler type"""
    # Define scaler mappings based on feature characteristics
    scaler_mapping = {
        "robust": ["log_atr"],  # Features with potential outliers
        "standard": [
            "close_log_return",
            "log_volume",
            "spread",
            "ret_mean",
            "rv",
            "sqrt_rv",
            "ema",
            "_slope",
            "atr",
            "_percent",
            "_adjusted_return",
            "bb_",
            "dc",
            "macd",
        ],
        "minmax": [
            "adx",
            "plus_di",
            "minus_di",
            "rsi",
            "stoch_k",
            "stoch_d",
        ],  # Bounded features
    }

    def categorize_column(col_name):
        """Categorize column by scaler type based on name patterns"""
        for scaler_type, patterns in scaler_mapping.items():
            if any(pattern in col_name for pattern in patterns):
                return scaler_type
        return "standard"  # Default fallback

    return scaler_mapping, categorize_column


def normalize_features(df, scaler_dir=None, file_prefix="", config=None):
    """Normalize features using appropriate scalers"""
    df = df.copy()

    # Get numeric columns for scaling (exclude timestamp, categorical)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "unix_time" in numeric_cols:
        numeric_cols.remove("unix_time")

    scaler_mapping, categorize_column = get_scaler_columns(config)

    # Group columns by scaler type
    cols_by_scaler = {"robust": [], "standard": [], "minmax": []}
    for col in numeric_cols:
        scaler_type = categorize_column(col)
        if col in df.columns:  # Double-check column exists
            cols_by_scaler[scaler_type].append(col)

    # Initialize and apply scalers
    scalers = {
        "robust": RobustScaler(),
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(feature_range=(-1, 1)),
    }

    for scaler_type, columns in cols_by_scaler.items():
        if columns:
            scaler = scalers[scaler_type]
            df[columns] = scaler.fit_transform(df[columns])

            # Save scaler if directory provided
            if scaler_dir:
                os.makedirs(scaler_dir, exist_ok=True)
                scaler_path = os.path.join(
                    scaler_dir, f"{file_prefix}_{scaler_type}_scaler.pkl"
                )
                joblib.dump(scaler, scaler_path)

    return df, scalers["robust"], scalers["standard"], scalers["minmax"]


def full_preprocessing_pipeline(input_path, output_dir, file_prefix=None, config=None):
    """Run complete preprocessing pipeline"""
    # Create output directories
    output_dir = Path(output_dir)
    processed_dir = output_dir / "processed"
    scalers_dir = output_dir / "scalers"

    for dir_path in [processed_dir, scalers_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(input_path)
    df = handle_timestamp_index(df)

    # Get feature flags from config
    features_config = get_feature_config(config)
    add_returns = features_config.get("add_returns", True)
    add_tech_indicators = features_config.get("add_technical_indicators", True)
    do_add_time_features = features_config.get("add_time_features", True)

    # Add features
    if add_returns:
        df = add_return_features(df, config=config)
    if add_tech_indicators:
        df = add_technical_indicators(df, config=config)

    # Add time features (need timestamp as column)
    df = handle_timestamp_column(df)
    if do_add_time_features:
        df = add_time_features(df)

    # Clean data
    df = df.dropna()

    # Determine output paths
    if config and file_prefix in [None, "processed"]:
        try:
            paths, sample_event, label_event = build_file_paths_from_config(config)
            processed_path = paths["processed"]
            normalized_path = paths["normalized"]
            scaler_prefix = f"{sample_event}_{label_event}"
        except ValueError:
            processed_path = processed_dir / "processed_data.pkl"
            normalized_path = processed_dir / "normalized_data.pkl"
            scaler_prefix = "scalers"
    else:
        processed_path = processed_dir / f"{file_prefix}_processed.pkl"
        normalized_path = processed_dir / f"{file_prefix}_normalized.pkl"
        scaler_prefix = file_prefix or "scalers"

    # Save processed data
    df_processed = handle_timestamp_index(df.copy())
    df_processed.to_pickle(processed_path)

    # Normalize and save
    df_normalized, *scalers = normalize_features(df, scalers_dir, scaler_prefix, config)
    df_normalized = handle_timestamp_index(df_normalized)
    df_normalized.to_pickle(normalized_path)

    return df_normalized


@click.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="YAML config file")
@click.option("--mode", "-m", type=click.Choice(["full", "features"]), default="full")
@click.option(
    "--input", "-i", "input_path", type=click.Path(exists=True), help="Input file"
)
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--prefix", "-p", default="processed", help="File prefix")
def main(config, mode, input_path, output, prefix):
    """Forex Data Preprocessing Pipeline"""
    # Load config
    config_data = load_config(config) if config else None

    # Determine input file
    if not input_path:
        if not config_data:
            raise click.ClickException("Input file required when no config provided")

        try:
            paths, _, _ = build_file_paths_from_config(config_data)
            input_path = paths["processed"]
        except ValueError as e:
            if "data" in config_data and "raw" in config_data["data"]:
                input_path = config_data["data"]["raw"]
            else:
                raise click.ClickException(f"Cannot determine input file: {e}")

    if not os.path.exists(input_path):
        raise click.ClickException(f"Input file not found: {input_path}")

    if mode == "full":
        output_dir = output or "data"
        full_preprocessing_pipeline(input_path, output_dir, prefix, config_data)

    elif mode == "features":
        # Features-only mode
        df = load_data(input_path)
        df = handle_timestamp_index(df)

        # Add features based on config
        features_config = get_feature_config(config_data)
        if features_config.get("add_returns", True):
            df = add_return_features(df, config=config_data)
        if features_config.get("add_technical_indicators", True):
            df = add_technical_indicators(df, config=config_data)

        df = handle_timestamp_column(df)
        if features_config.get("add_time_features", True):
            df = add_time_features(df)
        df = df.dropna()

        # Determine output path
        if output:
            output_path = output
        elif config_data:
            try:
                paths, _, _ = build_file_paths_from_config(config_data)
                output_path = paths["processed"]
            except ValueError:
                output_path = str(input_path).replace(".pkl", "_features.pkl")
        else:
            output_path = str(input_path).replace(".pkl", "_features.pkl")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = handle_timestamp_index(df)
        df.to_pickle(output_path)


if __name__ == "__main__":
    main()
