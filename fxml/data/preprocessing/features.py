import numpy as np
import pandas as pd
import pandas_ta as ta

from fxml.data.preprocessing.fractional_differentiation import (
    find_optimal_fraction,
    ts_differencing_tau,
)


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


def add_returns(df, config={}):
    """Add return features (delta, return, log return)"""
    print("Adding return features...")
    df = df.copy()

    price_cols = config.get("price_cols", ["close"])
    d_value = config.get("d_value", None)

    for col in price_cols:
        # Basic return features
        df[f"{col}_pct_return"] = df[col] / df[col].shift(1) - 1
        df[f"{col}_return"] = df[col] - df[col].shift(1)
        df[f"{col}_log_return"] = np.log(df[col] / df[col].shift(1))
        if d_value is not None:
            df[f"{col}_fd_return"] = ts_differencing_tau(df[col], 0.5, 1e-5)
        else:
            print("Finding optimal d_value...")
            d_value, optimal_fd = find_optimal_fraction(df[col], 1e-5, 0.01)
            df[f"{col}_fd_return"] = optimal_fd
        print(f"✓ Frational Fifferentiations, d-value: {d_value}")

    return df


def add_technical_indicators(df, config={}):
    """Add technical indicators"""
    print("Adding technical indicators...")
    df = df.copy()

    # RV (Realized Volatility)
    # rv_windows = config.get("rv_windows", [5, 10, 15, 20])
    # for window in rv_windows:
    #     df[f"rv{window}"] = df["close_log_return"].pow(2).rolling(window).sum()
    #     df[f"log_rv{window}"] = np.log1p(df[f"rv{window}"])
    #     df[f"sqrt_rv{window}"] = df[f"rv{window}"].pow(0.5)
    # print(f"  ✓ RV features for windows: {rv_windows}")

    # EMAs
    ema_windows = config.get("ema_windows", [5, 20])
    for window in ema_windows:
        df.ta.ema(length=window, append=True)
    print(f"  ✓ EMA features for windows: {ema_windows}")

    # ATR
    atr_windows = config.get("atr_windows", [14, 20])
    for window in atr_windows:
        df.ta.atr(length=window, append=True)
    print(f"  ✓ ATR features for windows: {atr_windows}")

    # ADX
    adx_windows = config.get("adx_windows", [14, 20])
    for window in adx_windows:
        df.ta.adx(length=window, append=True)
    print(f"  ✓ ADX features for windows: {adx_windows}")

    # RSI
    rsi_windows = config.get("rsi_windows", [14, 20])
    for window in rsi_windows:
        df.ta.rsi(length=window, append=True)
    print(f"  ✓ RSI features for windows: {rsi_windows}")

    # Bollinger Bands
    bb_windows = config.get("bb_windows", [14, 20])
    for window in bb_windows:
        df.ta.bbands(length=window, append=True)
    print(f"  ✓ BBands features for windows: {bb_windows}")

    # MACD
    macd_config = config.get("macd", [{"fast": 12, "slow": 26, "signal": 9}])
    for conf in macd_config:
        df.ta.macd(
            fast=conf["fast"],
            slow=conf["slow"],
            signal=conf["signal"],
            append=True,
        )
        print(
            f"  ✓ MACD (fast={conf['fast']}, slow={conf['slow']}, signal={conf['signal']})"
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
