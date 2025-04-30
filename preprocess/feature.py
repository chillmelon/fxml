import pandas as pd
import numpy as np
from tqdm import tqdm

def add_delta(df, delta_columns=['close'], show_progress=True):
    """
    Add value delta to dataframe

    Args:
        df (pandas.DataFrame): The dataframe to add indicators to.
        config (dict): Configuration dictionary.
        show_progress (bool, optional): Whether to show progress bar. Defaults to True.

    Returns:
        pandas.DataFrame: The dataframe with added delta values.
    """
    df = df.copy()

    # Create progress bar if show_progress is True
    progress_iter = (
        tqdm(delta_columns, desc="Adding price deltas")
        if show_progress
        else delta_columns
    )

    for col in progress_iter:
        # Calculate absolute delta (change from previous row)
        df[f"{col}_delta"] = df[col].diff()

        # Calculate percentage delta
        df[f"{col}_pct_delta"] = df[col].pct_change() * 100

    # Fill NaN values
    df.bfill()

    return df

def add_direction(df, delta_columns=['close'], threshold=0.005, show_progress=True):
    """
    Add value delta to dataframe

    Args:
        df (pandas.DataFrame): The dataframe to add indicators to.
        config (dict): Configuration dictionary.
        show_progress (bool, optional): Whether to show progress bar. Defaults to True.

    Returns:
        pandas.DataFrame: The dataframe with added delta values.
    """
    df = df.copy()

    # Create progress bar if show_progress is True
    progress_iter = (
        tqdm(delta_columns, desc="Adding price direction")
        if show_progress
        else delta_columns
    )

    for col in progress_iter:
        # Calculate absolute delta (change from previous row)
        df[f"{col}_delta"] = df[col].diff()

        # Apply threshold to determine direction
        df[f"{col}_direction"] = df[f"{col}_delta"].apply(
            lambda x: 1 if x > threshold else (-1 if x < -threshold else 0)
        )
    # Fill NaN values
    df.bfill()

    return df

def add_technical_indicators(df, window=14, show_progress=True):
    """
    Add technical indicators to the dataframe.

    Args:
        df (pandas.DataFrame): The dataframe to add indicators to.
        window (int, optional): The window size for indicators. Defaults to 14.
        show_progress (bool, optional): Whether to show progress bar. Defaults to True.

    Returns:
        pandas.DataFrame: The dataframe with added indicators.
    """
    df = df.copy()

    # Create progress bar for technical indicators calculation
    indicators = [
        "Moving Averages",
        "Exponential Moving Averages",
        "RSI",
        "MACD",
        "Bollinger Bands",
        "ATR",
    ]

    progress_iter = (
        tqdm(indicators, desc="Adding technical indicators")
        if show_progress
        else indicators
    )

    for indicator in progress_iter:
        if indicator == "Moving Averages":
            # Moving Averages
            df["ma7"] = df["close"].rolling(window=7).mean()
            df["ma14"] = df["close"].rolling(window=14).mean()
            df["ma30"] = df["close"].rolling(window=30).mean()

        elif indicator == "Exponential Moving Averages":
            # Exponential Moving Averages
            df["ema7"] = df["close"].ewm(span=7).mean()
            df["ema14"] = df["close"].ewm(span=14).mean()
            df["ema30"] = df["close"].ewm(span=30).mean()

        elif indicator == "RSI":
            # Relative Strength Index (RSI)
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))

        elif indicator == "MACD":
            # Moving Average Convergence Divergence (MACD)
            ema12 = df["close"].ewm(span=12).mean()
            ema26 = df["close"].ewm(span=26).mean()
            df["macd"] = ema12 - ema26
            df["macd_signal"] = df["macd"].ewm(span=9).mean()
            df["macd_hist"] = df["macd"] - df["macd_signal"]

        elif indicator == "Bollinger Bands":
            # Bollinger Bands
            df["bb_middle"] = df["close"].rolling(window=window).mean()
            df["bb_std"] = df["close"].rolling(window=window).std()
            df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * 2)
            df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * 2)

        elif indicator == "ATR":
            # Average True Range (ATR)
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift()).abs()
            low_close = (df["low"] - df["close"].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df["atr"] = true_range.rolling(window=window).mean()

    # Fill NaN values
    df.bfill()

    return df


def create_time_features(df, date_column="date"):
    """
    Create time-based features from datetime column.

    Args:
        df (pandas.DataFrame): The dataframe to add features to.
        date_column (str, optional): The name of the date column. Defaults to 'date'.

    Returns:
        pandas.DataFrame: The dataframe with added time features.
    """
    df = df.copy()

    # Convert to datetime if needed
    if df[date_column].dtype != "datetime64[ns]":
        df[date_column] = pd.to_datetime(df[date_column])

    # Extract time features
    df["hour"] = df[date_column].dt.hour
    df["day"] = df[date_column].dt.day
    df["day_of_week"] = df[date_column].dt.dayofweek
    df["week"] = df[date_column].dt.isocalendar().week
    df["month"] = df[date_column].dt.month
    df["quarter"] = df[date_column].dt.quarter
    df["year"] = df[date_column].dt.year

    # Cyclical features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df
