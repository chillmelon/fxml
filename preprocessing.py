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
import pandas as pd

from libs.feature_engineering import (
    add_return_features,
    add_technical_indicators,
    add_time_features,
    handle_timestamp_column,
    handle_timestamp_index,
)
from libs.normalization import normalize_features
from libs.utils import build_file_paths_from_config, load_config


def load_data(file_path):
    """Load data from pickle or CSV file"""
    print(f"Loading data from: {file_path}")

    if str(file_path).endswith(".pkl"):
        df = pd.read_pickle(file_path)
        print(f"✓ Loaded pickle data: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
    elif str(file_path).endswith(".csv"):
        df = pd.read_csv(file_path)
        print(f"✓ Loaded CSV data: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
    else:
        raise ValueError("Unsupported file format. Please use .pkl or .csv")


def full_preprocessing_pipeline(input_path, output_dir, file_prefix=None, config=None):
    """Run complete preprocessing pipeline"""

    print("=" * 60)
    print("FOREX DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    # Create output directories
    output_dir = Path(output_dir)
    processed_dir = output_dir / "processed"
    scalers_dir = output_dir / "scalers"

    for dir_path in [processed_dir, scalers_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(input_path)
    df = handle_timestamp_index(df)
    print(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Get feature flags from config
    features_config = (
        config.get("preprocessing", {}).get("features", {}) if config else {}
    )
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
    rows_before = df.shape[0]
    df = df.dropna()
    rows_after = df.shape[0]
    print(f"Data cleaning: dropped {rows_before - rows_after:,} rows with NaN values")
    print(f"Final dataset: {rows_after:,} rows, {df.shape[1]} columns")

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
    print(f"Saved processed data to: {processed_path}")

    # Normalize and save
    df_normalized, *scalers = normalize_features(df, scalers_dir, scaler_prefix, config)
    df_normalized = handle_timestamp_index(df_normalized)
    df_normalized.to_pickle(normalized_path)
    print(f"Saved normalized data to: {normalized_path}")

    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print(
        f"✓ Final output: {df_normalized.shape[0]:,} rows, {df_normalized.shape[1]} columns"
    )
    print("=" * 60)

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
    click.echo("\n" + "=" * 60)
    click.echo("FOREX DATA PREPROCESSING PIPELINE")
    click.echo("=" * 60)
    click.echo(f"Mode: {mode}")

    # Load config
    config_data = load_config(config) if config else None
    if config:
        click.echo(f"Config loaded from: {config}")
    else:
        click.echo("No config file provided - using defaults")

    # Determine input file
    if not input_path:
        if not config_data:
            raise click.ClickException("Input file required when no config provided")

        try:
            paths, _, _ = build_file_paths_from_config(config_data)
            input_path = paths["resampled"]
        except ValueError as e:
            if "data" in config_data and "raw" in config_data["data"]:
                input_path = config_data["data"]["raw"]
            else:
                raise click.ClickException(f"Cannot determine input file: {e}")

    if not os.path.exists(input_path):
        raise click.ClickException(f"Input file not found: {input_path}")

    click.echo(f"Input file: {input_path}")

    if mode == "full":
        output_dir = output or "data"
        click.echo(f"Output directory: {output_dir}")
        click.echo("")
        full_preprocessing_pipeline(input_path, output_dir, prefix, config_data)

    elif mode == "features":
        click.echo("Running features-only mode...")
        click.echo("")

        # Features-only mode
        df = load_data(input_path)
        df = handle_timestamp_index(df)
        click.echo(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

        # Add features based on config
        features_config = (
            config_data.get("preprocessing", {}).get("features", {})
            if config_data
            else {}
        )
        if features_config.get("add_returns", True):
            df = add_return_features(df, config=config_data)
        if features_config.get("add_technical_indicators", True):
            df = add_technical_indicators(df, config=config_data)

        df = handle_timestamp_column(df)
        if features_config.get("add_time_features", True):
            df = add_time_features(df)

        rows_before = df.shape[0]
        df = df.dropna()
        rows_after = df.shape[0]
        if rows_before != rows_after:
            click.echo(f"Dropped {rows_before - rows_after:,} rows with NaN values")

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
        click.echo(f"\n✓ Features saved to: {output_path}")
        click.echo(f"Final dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    main()
