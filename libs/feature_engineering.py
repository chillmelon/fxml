#!/usr/bin/env python
# coding: utf-8

"""
Feature Engineering Module

This module contains all feature engineering functions for the forex trading pipeline.
Includes return features, technical indicators, and time-based features.
"""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands, DonchianChannel


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


def add_return_features(df, price_col="close", config=None):
    """Add return features (delta, return, log return)"""
    print("Adding return features...")
    df = df.copy()

    # Basic return features
    df[f"{price_col}_delta"] = df[price_col] - df[price_col].shift(1)
    df[f"{price_col}_return"] = df[price_col] / df[price_col].shift(1) - 1
    df[f"{price_col}_log_return"] = np.log(df[price_col] / df[price_col].shift(1))

    return_config = config.get("preprocessing", {}).get("features", {}).get("return_features", {}) if config else {}
    rolling_windows = return_config.get("rolling_mean_windows", [5, 10])

    # Add rolling return means
    for window in rolling_windows:
        df[f"ret_mean_{window}"] = (
            df[f"{price_col}_log_return"]
            .rolling(window=window, min_periods=window)
            .mean()
        )
    if rolling_windows:
        print(f"  ✓ Added rolling return means for windows: {rolling_windows}")

    # Add log volume
    if return_config.get("log_volume", True) and "volume" in df.columns:
        df["log_volume"] = np.log1p(df["volume"])
        print(f"  ✓ Added log volume")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def add_technical_indicators(df, config=None):
    """Add technical indicators"""
    print("Adding technical indicators...")
    df = df.copy()
    ta_config = config.get("preprocessing", {}).get("features", {}).get("technical_indicators", {}) if config else {}

    # RV (Realized Volatility)
    rv_windows = ta_config.get("rv_windows", [5, 10, 15, 20])
    for window in rv_windows:
        df[f"rv{window}"] = df["close_log_return"].pow(2).rolling(window).sum()
        df[f"log_rv{window}"] = np.log1p(df[f"rv{window}"])
        df[f"sqrt_rv{window}"] = df[f"rv{window}"].pow(0.5)
    print(f"  ✓ RV features for windows: {rv_windows}")

    # EMAs
    ema_windows = ta_config.get("ema_windows", [5, 20])
    for window in ema_windows:
        ema = EMAIndicator(close=df["close"], window=window)
        df[f"ema{window}"] = ema.ema_indicator()
        df[f"ema{window}_slope"] = df[f"ema{window}"].diff()
        df[f"close_above_ema{window}"] = (df["close"] > df[f"ema{window}"]).astype(int)
    print(f"  ✓ EMA features for windows: {ema_windows}")

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
    rows_before = df.shape[0]
    df.dropna(inplace=True)
    rows_after = df.shape[0]
    if rows_before != rows_after:
        print(f"  → Dropped {rows_before - rows_after} rows with ATR NaN values")

    for window in atr_windows:
        df[f"log_atr{window}"] = np.log1p(df[f"atr{window}"])
        df[f"atr{window}_percent"] = df[f"atr{window}"] / df["close"]
        df[f"atr{window}_adjusted_return"] = df["close_delta"] / df[f"atr{window}"]
    print(f"  ✓ ATR features for windows: {atr_windows}")

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
    print(f"  ✓ ADX features for windows: {adx_windows}")

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
    print(
        f"  ✓ Bollinger Bands (window={bb_config['window']}, dev={bb_config['window_dev']})"
    )

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
    print(f"  ✓ Donchian Channel features for windows: {dc_windows}")

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
    print(
        f"  ✓ Stochastic Oscillator (window={stoch_config['window']}, smooth={stoch_config['smooth_window']})"
    )

    # RSI
    rsi_windows = ta_config.get("rsi_windows", [14, 20])
    for window in rsi_windows:
        rsi = RSIIndicator(close=df["close"], window=window)
        df[f"rsi{window}"] = rsi.rsi()
        df[f"rsi{window}_slope"] = df[f"rsi{window}"].diff()
    print(f"  ✓ RSI features for windows: {rsi_windows}")

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
    print(
        f"  ✓ MACD (slow={macd_config['window_slow']}, fast={macd_config['window_fast']}, signal={macd_config['window_sign']})"
    )
    return df


def add_time_features(df, timestamp_col="timestamp"):
    """Add cyclical time features"""
    print("Adding time features...")
    df = df.copy()

    # Ensure timestamp is datetime
    if df[timestamp_col].dtype != "datetime64[ns]":
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        print(f"  ✓ Converted {timestamp_col} to datetime")

    # Unix time
    df["unix_time"] = df[timestamp_col].astype("int64") / 1e9
    print(f"  ✓ Added unix_time")

    # Extract components
    df["hour"] = df[timestamp_col].dt.hour
    df["dow"] = df[timestamp_col].dt.dayofweek
    df["dom"] = df[timestamp_col].dt.day
    df["month"] = df[timestamp_col].dt.month
    print(f"  ✓ Extracted time components (hour, dow, dom, month)")

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["dom_sin"] = np.sin(2 * np.pi * df["dom"] / 31)
    df["dom_cos"] = np.cos(2 * np.pi * df["dom"] / 31)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    print(f"  ✓ Added cyclical encodings (sin/cos pairs)")

    return df