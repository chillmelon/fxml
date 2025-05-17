#!/usr/bin/env python
# coding: utf-8

"""
Forex Data Preprocessing Pipeline

This script processes raw forex time series data by:
1. Loading the data
2. Handling missing values
3. Identifying time gaps and grouping continuous segments with configurable tolerance
4. Engineering features (delta, returns, movement direction)
5. Encoding target variables
6. Filtering time groups by minimum length
7. Saving the processed data

Usage:
    python forex_preprocessing.py --input <input_file_path> --output <output_file_path> --time_gap_tolerance <seconds>
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib


def load_data(file_path):
    """Load data from pickle or CSV file"""
    print(f"Loading data from {file_path}")

    if file_path.endswith('.pkl'):
        df = pd.read_pickle(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .pkl or .csv")

    print(f"Data loaded with shape: {df.shape}")
    return df


def handle_missing_values(df):
    """Handle missing values in the dataset"""
    print("Handling missing values...")

    # Check for missing values
    na_count_before = df.isna().sum()
    print(f"Missing values before cleaning:\n{na_count_before}")

    # Drop rows with missing values
    df = df.dropna()

    # Verify missing values are gone
    na_count_after = df.isna().sum()
    print(f"Missing values after cleaning:\n{na_count_after}")
    print(f"Data shape after cleaning: {df.shape}")

    return df


def identify_time_groups(df, time_gap_tolerance=60):
    """
    Identify continuous time segments and assign time group labels

    Args:
        df: DataFrame with a timestamp column
        time_gap_tolerance: Maximum acceptable time gap in seconds between
                           consecutive data points to be considered part of the same group

    Returns:
        DataFrame with time_group column added
    """
    print(f"Identifying time groups based on time continuity (gap tolerance: {time_gap_tolerance}s)...")

    # Ensure df is a copy (not a view) to avoid SettingWithCopyWarning
    df = df.copy()

    # Convert timestamp to datetime if not already
    df.loc[:, 'timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Calculate time difference between consecutive rows
    df.loc[:, 'time_delta'] = df['timestamp'].diff().dt.total_seconds()

    # Create time groups based on time gaps
    df.loc[:, 'time_group'] = (df['time_delta'] > time_gap_tolerance).cumsum().astype(int)

    # Drop temporary column
    df = df.drop(columns='time_delta')

    # Summarize time groups
    group_counts = df['time_group'].value_counts()
    print(f"Created {len(group_counts)} time groups")
    print(f"Min group size: {group_counts.min()}, Max group size: {group_counts.max()}")

    return df


def add_delta_and_returns(df, price_col='close', group_col='time_group'):
    """Add price delta and return features within each time group"""
    print("Adding price deltas and returns...")

    df = df.copy()

    def calc(group):
        group[f"{price_col}_delta"] = group[price_col] - group[price_col].shift(1)
        group[f"{price_col}_return"] = group[price_col] / group[price_col].shift(1) - 1
        return group

    df = df.groupby(group_col, group_keys=False).apply(calc)

    # Replace infinities with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Reset index for consistency
    df = df.reset_index(drop=True)

    # Drop rows with NaN values created by calculating deltas
    na_count = df.isna().sum()
    print(f"NaN values after delta calculation:\n{na_count}")
    df = df.dropna()
    print(f"Data shape after dropna: {df.shape}")

    return df


def add_direction_labels(df, delta_columns=['close'], threshold=3e-5):
    """Add directional movement labels based on returns and threshold"""
    print(f"Adding direction labels with threshold {threshold}...")

    df = df.copy()

    for col in delta_columns:
        df[f"{col}_direction"] = df[f"{col}_return"].apply(
            lambda x: 'up' if x > threshold else ('down' if x < -threshold else 'flat')
        )

    # Show distribution of direction labels
    direction_counts = df[f"{delta_columns[0]}_direction"].value_counts()
    print(f"Direction distribution:\n{direction_counts}")

    return df


def encode_labels(df, class_col='close_direction'):
    """Encode direction labels to numeric values and one-hot encoding"""
    print("Encoding direction labels...")

    df = df.copy()

    # One-hot encode with get_dummies
    one_hot = pd.get_dummies(df[class_col], prefix='prob').astype('float32')
    df = df.join(one_hot)

    # Label encode for categorical target
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df[class_col])

    direction_counts = df["label"].value_counts()
    print(f"Encoded label distribution:\n{direction_counts}")
    print(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

    return df, label_encoder


def filter_time_groups(df, min_length=31):
    """Filter time groups by minimum length requirement"""
    print(f"Filtering time groups with minimum length {min_length}...")

    before_count = df['time_group'].nunique()

    # Keep only groups with at least min_length rows
    df = df.groupby("time_group").filter(lambda g: len(g) >= min_length)

    after_count = df['time_group'].nunique()
    print(f"Time groups before filter: {before_count}")
    print(f"Time groups after filter: {after_count}")
    print(f"Data shape after filtering: {df.shape}")

    return df


# Plotting function removed as requested


def main():
    parser = argparse.ArgumentParser(description='Forex Data Preprocessing Pipeline')
    parser.add_argument('--input', type=str, required=True, help='Input file path (.pkl or .csv)')
    parser.add_argument('--output', type=str, required=True, help='Output pickle file path')
    parser.add_argument('--encoder_output', type=str, help='Path to save label encoder')
    parser.add_argument('--seq_len', type=int, default=30, help='Sequence length for filtering')
    parser.add_argument('--horizon', type=int, default=1, help='Prediction horizon for filtering')
    parser.add_argument('--threshold', type=float, default=3e-5,
                        help='Threshold for price movement direction classification')
    parser.add_argument('--time_gap_tolerance', type=int, default=60,
                        help='Maximum time gap in seconds between consecutive data points to be considered part of the same group')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Define minimum length requirement for time groups
    min_len = args.seq_len + args.horizon

    # Set encoder output path if not provided
    if not args.encoder_output:
        args.encoder_output = os.path.join(
            os.path.dirname(args.output),
            'label_encoder.pkl'
        )

    print("=" * 50)
    print(f"Starting Forex Data Preprocessing Pipeline")
    print("=" * 50)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Encoder output: {args.encoder_output}")
    print(f"Minimum time group length: {min_len}")
    print(f"Time gap tolerance: {args.time_gap_tolerance}s")
    print("=" * 50)

    # Load data
    df = load_data(args.input)

    # Handle missing values
    df = handle_missing_values(df)

    # Identify time groups with configurable time gap tolerance
    df = identify_time_groups(df, time_gap_tolerance=args.time_gap_tolerance)

    # Add delta and return features
    df = add_delta_and_returns(df)

    # Add direction labels
    df = add_direction_labels(df, threshold=args.threshold)

    # Encode labels
    df, encoder = encode_labels(df)

    # Filter time groups by minimum length
    df = filter_time_groups(df, min_length=min_len)

    # Save processed data
    print(f"Saving processed data to {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_pickle(args.output)

    # Save label encoder
    print(f"Saving label encoder to {args.encoder_output}")
    os.makedirs(os.path.dirname(args.encoder_output), exist_ok=True)
    joblib.dump(encoder, args.encoder_output)

    print("=" * 50)
    print("Preprocessing pipeline completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
