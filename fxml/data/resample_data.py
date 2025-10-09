import pandas as pd

from fxml.utils import load_config


def filter_data_by_date_range(df, start_date=None, end_date=None):
    """Filter dataframe by date range

    Args:
        df: DataFrame with timestamp column
        start_date: Start date as string (yyyymmdd) or datetime
        end_date: End date as string (yyyymmdd) or datetime

    Returns:
        Filtered dataframe
    """
    if start_date is None and end_date is None:
        return df

    df = df.copy()

    # Convert timestamp to datetime if not already
    if "timestamp" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    else:
        raise ValueError("DataFrame must have a 'timestamp' column")

    # Convert date strings to datetime objects
    start_date = pd.to_datetime(start_date) if start_date is not None else None
    end_date = pd.to_datetime(end_date) if end_date is not None else None

    # Apply date filtering
    original_shape = df.shape[0]

    if start_date is not None:
        df = df[df["timestamp"] >= start_date]

    if end_date is not None:
        df = df[df["timestamp"] <= end_date]

    filtered_shape = df.shape[0]
    print(
        f"Date filtering: {original_shape:,} -> {filtered_shape:,} rows ({filtered_shape/original_shape*100:.1f}% retained)"
    )

    if filtered_shape == 0:
        print("WARNING: No data remains after date filtering. Check your date range.")

    return df


def create_time_bar(df, minutes: int):
    """Resample tick data to time-based OHLCV bars"""
    print(f"Resampling to {minutes}-minute bars...")

    df = df[["timestamp", "askPrice", "bidPrice", "askVolume", "bidVolume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["mid"] = (df["askPrice"] + df["bidPrice"]) / 2
    # Volume is in millions, convert to actual volume
    df["volume"] = (df["askVolume"] + df["bidVolume"]) * 1_000_000
    df["spread"] = df["askPrice"] - df["bidPrice"]

    df.set_index("timestamp", inplace=True)
    df = df[["mid", "volume", "spread"]]

    ohlcv = df.resample(f"{minutes}min").agg(
        {"mid": ["first", "max", "min", "last"], "volume": "sum", "spread": "mean"}
    )

    ohlcv.columns = ["open", "high", "low", "close", "volume", "spread"]
    ohlcv = ohlcv.dropna(subset=["open"])
    ohlcv.reset_index(inplace=True)

    print(f"Resampled data shape: {ohlcv.shape}")
    return ohlcv


def main():
    print("=" * 60)
    print("FOREX DATA RESAMPLING PIPELINE")
    print("=" * 60)

    config = load_config("configs/resample.yaml")
    raw_path = config.get("data", {}).get("raw", {})

    symbol = config.get("symbol", "USDJPY")
    minutes = config.get("resampling", {}).get("minutes", [5])
    date_ranges = config.get("resampling", {}).get("date", [])

    print(f"Loading data from {raw_path}")
    df = pd.read_csv(raw_path)

    for date_range in date_ranges:
        date_from = str(date_range.get("from", None))
        date_to = str(date_range.get("to", None))
        df_filtered = filter_data_by_date_range(df, date_from, date_to)

        for minute in minutes:
            df_bar = create_time_bar(df_filtered, minute)
            RESAMPLED_NAME = f"{symbol}-{minute}m-{date_from}-{date_to}"
            df_bar.to_pickle(f"data/resampled/{RESAMPLED_NAME}.pkl")
            print(f"Resampled data saved to {RESAMPLED_NAME}.pkl")

    print("=" * 60)
    print("RESAMPLING COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
