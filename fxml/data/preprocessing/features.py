import numpy as np
import pandas as pd

from src.utils import load_config


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


def add_differences(df):
    df_log = np.log(df["close"])
    df["return"] = df["close"] - df["close"].shift(1)
    df["log_return"] = df_log["close"] - df_log["close"].shift(1)
    return


def add_return_features(df, price_col="close", config=None):
    """Add return features (delta, return, log return)"""
    print("Adding return features...")
    df = df.copy()

    # Basic return features
    df[f"{price_col}_delta"] = df[price_col] - df[price_col].shift(1)
    df[f"{price_col}_return"] = df[price_col] / df[price_col].shift(1) - 1
    df[f"{price_col}_log_return"] = np.log(df[price_col] / df[price_col].shift(1))

    return_config = (
        config.get("preprocessing", {}).get("features", {}).get("return_features", {})
        if config
        else {}
    )
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
    ta_config = (
        config.get("preprocessing", {})
        .get("features", {})
        .get("technical_indicators", {})
        if config
        else {}
    )

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
        df.ta.ema(length=window, append=True)
    print(f"  ✓ EMA features for windows: {ema_windows}")

    # ATR
    atr_windows = ta_config.get("atr_windows", [14, 20])
    for window in atr_windows:
        df.ta.atr(length=window, append=True)

    # ADX
    adx_windows = ta_config.get("adx_windows", [14, 20])
    for window in adx_windows:
        df.ta.adx(length=window, append=True)
    print(f"  ✓ ADX features for windows: {adx_windows}")

    # RSI
    rsi_windows = ta_config.get("rsi_windows", [14, 20])
    for window in rsi_windows:
        df.ta.rsi(length=window, append=True)
    print(f"  ✓ RSI features for windows: {rsi_windows}")

    # MACD
    macd_config = ta_config.get("macd", {"fast": 12, "slow": 26, "signal": 9})
    df.ta.macd(
        fast=macd_config["fast"],
        slow=macd_config["slow"],
        signal=macd_config["signal"],
    )
    print(
        f"  ✓ MACD (fast={macd_config['fast']}, slow={macd_config['slow']}, signal={macd_config['signal']})"
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
