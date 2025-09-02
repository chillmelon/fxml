#!/usr/bin/env python
# coding: utf-8

"""
Normalization Module

This module contains all data normalization and scaling functions for the forex trading pipeline.
Includes scaler selection, feature normalization, and scaler persistence.
"""

import os
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


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
    print("Normalizing features...")
    df = df.copy()

    # Get numeric columns for scaling (exclude timestamp, categorical)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "unix_time" in numeric_cols:
        numeric_cols.remove("unix_time")

    _, categorize_column = get_scaler_columns(config)

    # Group columns by scaler type
    cols_by_scaler = {"robust": [], "standard": [], "minmax": []}
    for col in numeric_cols:
        scaler_type = categorize_column(col)
        if col in df.columns:  # Double-check column exists
            cols_by_scaler[scaler_type].append(col)

    for scaler_type, columns in cols_by_scaler.items():
        if columns:
            print(f"  ✓ {scaler_type.upper()} scaler: {len(columns)} columns")

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
                print(f"    → Saved {scaler_type} scaler to: {scaler_path}")

    return df, scalers["robust"], scalers["standard"], scalers["minmax"]