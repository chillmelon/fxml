import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def add_delta(df, price_col: str = 'close') -> pd.DataFrame:
    """
    Adds delta features to a price dataframe.

    - Adds 'close_delta': price difference between t and t-1
    - Adds 'close_return': percentage return between t and t-1

    Args:
        df (pd.DataFrame): Input dataframe with a 'close' column
        price_col (str): Column to compute deltas on (default 'close')

    Returns:
        pd.DataFrame: Copy of dataframe with new delta features added
    """
    df = df.copy()

    # Shifted previous prices
    prev = df[price_col].shift(1)

    # Absolute change: ΔP = P[t] - P[t-1]
    df[f"{price_col}_delta"] = df[price_col] - prev

    # % return: (P[t] - P[t-1]) / P[t-1]
    # Equivalent to: (P[t] / P[t-1]) - 1
    df[f"{price_col}_return"] = df[price_col] / prev - 1

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df.bfill().reset_index(drop=True)


def add_direction(df, delta_columns=['close'], threshold=0.005):
    """
    Add directional class labels based on deltas and a threshold.
    """
    df = df.copy()
    print(f"[INFO] Adding direction columns for: {delta_columns} with threshold {threshold}")
    for col in delta_columns:
        df[f"{col}_direction"] =  df[f"{col}_return"].apply(lambda x: 'up' if x > threshold else ('down' if x < -threshold else 'flat'))
        label_encoder = LabelEncoder()
        df["label"] =  label_encoder.fit_transform(df[f"{col}_direction"])
        direction_counts = df["label"].value_counts()
        print(direction_counts)
        print(label_encoder.classes_)
    return df.bfill()

def add_technical_indicators(df, window=14):
    """
    Add common technical indicators to the DataFrame.
    """
    df = df.copy()
    print(f"[INFO] Adding technical indicators with window={window}")

    # Moving Averages
    print("[INFO] → Moving Averages")
    df["ma7"] = df["close"].rolling(window=7).mean()
    df["ma14"] = df["close"].rolling(window=14).mean()
    df["ma30"] = df["close"].rolling(window=30).mean()

    # Exponential Moving Averages
    print("[INFO] → Exponential Moving Averages")
    df["ema7"] = df["close"].ewm(span=7).mean()
    df["ema14"] = df["close"].ewm(span=14).mean()
    df["ema30"] = df["close"].ewm(span=30).mean()

    # RSI
    print("[INFO] → RSI")
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    print("[INFO] → MACD")
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    print("[INFO] → Bollinger Bands")
    df["bb_middle"] = df["close"].rolling(window=window).mean()
    df["bb_std"] = df["close"].rolling(window=window).std()
    df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * 2)
    df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * 2)

    # ATR
    print("[INFO] → ATR")
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df["atr"] = true_range.rolling(window=window).mean()

    return df.bfill()


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
