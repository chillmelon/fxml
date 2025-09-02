#!/usr/bin/env python
# coding: utf-8

"""
Forex Data Resampling Pipeline

This script handles resampling of tick data to time-based or event-based bars.
Supports both time-based OHLCV bars and dollar bars based on configuration.

Usage:
    python resampling.py --config config/config.yaml
    python resampling.py --config config/config.yaml --input data/raw/tick_data.csv --output data/resampled/
    python resampling.py --input data/raw/tick_data.csv --type time --minutes 5 --symbol USDJPY --output data/resampled/
    python resampling.py --input data/raw/tick_data.csv --type dollar --threshold 58000000 --symbol EURUSD --output data/resampled/
    python resampling.py --input data/raw/tick_data.csv --type time --minutes 5 --start-date 20230101 --end-date 20231231 --symbol USDJPY
    python resampling.py --input data/raw/tick_data.csv --type dollar --threshold 58000000 --start-date 20230601 --end-date 20230630 --symbol GBPUSD

Config file date range example:
resampling:
  type: dollar
  threshold: 58000000
  symbol: "USDJPY"
  date_range:
    start_date: "20230101"  # Format: YYYYMMDD
    end_date: "20231231"    # Format: YYYYMMDD
  output:
    dir: data/resampled
    # Output filename will be: USDJPY-58m-dollar-20230101-20231231.pkl
"""

import os

import click
import pandas as pd

from libs.utils import load_config, load_data, normalize_date_format


def filter_data_by_date_range(df, start_date=None, end_date=None):
    """Filter dataframe by date range

    Args:
        df: DataFrame with timestamp column
        start_date: Start date as string (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS) or datetime
        end_date: End date as string (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS) or datetime

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
    if start_date is not None:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        print(f"Filtering data from: {start_date}")

    if end_date is not None:
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        print(f"Filtering data to: {end_date}")

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


def resample_to_time_bar(df, minutes: int):
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


def resample_to_dollar_bar(df, threshold: float):
    """Resample tick data to dollar bars"""
    print(f"Resampling to dollar bars with threshold: ${threshold:,.0f}...")

    df = df[["timestamp", "askPrice", "bidPrice", "askVolume", "bidVolume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["mid"] = (df["askPrice"] + df["bidPrice"]) / 2
    # Volume is in millions, convert to actual volume
    df["volume"] = (df["askVolume"] + df["bidVolume"]) * 1_000_000
    df["dollar"] = df["mid"] * df["volume"]
    df["spread"] = df["askPrice"] - df["bidPrice"]

    bars = []
    cum_dollar = 0.0
    bar = {
        "open": None,
        "high": -float("inf"),
        "low": float("inf"),
        "close": None,
        "volume": 0.0,
        "spread_sum": 0.0,
        "count": 0,
        "start_time": None,
        "end_time": None,
    }

    for row in df.itertuples():
        price = row.mid
        vol = row.volume
        dol = row.dollar
        spread = row.spread
        ts = row.timestamp

        if bar["open"] is None:
            bar["open"] = price
            bar["start_time"] = ts

        bar["high"] = max(bar["high"], price)
        bar["low"] = min(bar["low"], price)
        bar["close"] = price
        bar["volume"] += vol
        bar["spread_sum"] += spread
        bar["count"] += 1
        bar["end_time"] = ts
        cum_dollar += dol

        if cum_dollar >= threshold:
            bars.append(
                {
                    "timestamp": bar["end_time"],
                    "open": bar["open"],
                    "high": bar["high"],
                    "low": bar["low"],
                    "close": bar["close"],
                    "volume": bar["volume"],
                    "spread": (
                        bar["spread_sum"] / bar["count"] if bar["count"] > 0 else None
                    ),
                }
            )
            cum_dollar = 0.0
            bar = {
                "open": None,
                "high": -float("inf"),
                "low": float("inf"),
                "close": None,
                "volume": 0.0,
                "spread_sum": 0.0,
                "count": 0,
                "start_time": None,
                "end_time": None,
            }

    result = pd.DataFrame(bars)
    print(f"Dollar bars shape: {result.shape}")
    return result


def resample_data(
    config=None,
    input_path=None,
    output_path=None,
    resample_type=None,
    minutes=None,
    threshold=None,
    start_date=None,
    end_date=None,
    symbol=None,
):
    """Main resampling function that handles both time and dollar bars"""

    print("=" * 60)
    print("FOREX DATA RESAMPLING PIPELINE")
    print("=" * 60)

    # Determine input file
    if not input_path:
        if config and "data" in config and "raw" in config["data"]:
            input_path = config["data"]["raw"]
        else:
            raise ValueError(
                "Input file must be specified via --input or in config file"
            )

    # Determine resampling parameters from config or arguments
    if not resample_type:
        if config and "resampling" in config:
            resample_type = config["resampling"].get("type", "time")
        else:
            resample_type = "time"

    if not minutes and resample_type == "time":
        if config and "resampling" in config:
            minutes = config["resampling"].get("minutes", 5)
        else:
            minutes = 5

    if not threshold and resample_type == "dollar":
        if config and "resampling" in config:
            threshold = config["resampling"].get("threshold", 58000000)
        else:
            threshold = 58000000

    # Get date range from config if not provided via arguments
    if not start_date and not end_date:
        if config and "resampling" in config and "date_range" in config["resampling"]:
            date_range = config["resampling"]["date_range"]
            start_date = date_range.get("start_date")
            end_date = date_range.get("end_date")
            if start_date or end_date:
                print(f"Using date range from config: {start_date} to {end_date}")

    # Get symbol from config if not provided via arguments
    if not symbol:
        if config and "resampling" in config:
            symbol = config["resampling"].get("symbol", "UNKNOWN")
        else:
            symbol = "UNKNOWN"

    # Load data
    df = load_data(input_path)

    # Apply date filtering if specified (from config or command line)
    df = filter_data_by_date_range(df, start_date, end_date)

    # Perform resampling
    if resample_type == "time":
        if not minutes:
            raise ValueError("--minutes required for time-based resampling")
        result = resample_to_time_bar(df, minutes)
        output_suffix = f"{minutes}m"
    elif resample_type == "dollar":
        if not threshold:
            raise ValueError("--threshold required for dollar bar resampling")
        result = resample_to_dollar_bar(df, threshold)
        # Convert threshold to readable format (e.g., 58000000 -> 58m)
        if threshold >= 1000000:
            threshold_str = f"{int(threshold/1000000)}m"
        elif threshold >= 1000:
            threshold_str = f"{int(threshold/1000)}k"
        else:
            threshold_str = str(int(threshold))
        output_suffix = f"{threshold_str}-dollar"
    else:
        raise ValueError("--type must be 'time' or 'dollar'")

    # Generate filename with symbol and date range
    date_suffix = ""
    if start_date or end_date:
        start_str = normalize_date_format(start_date) if start_date else ""
        end_str = normalize_date_format(end_date) if end_date else ""
        if start_str and end_str:
            date_suffix = f"-{start_str}-{end_str}"
        elif start_str:
            date_suffix = f"-from{start_str}"
        elif end_str:
            date_suffix = f"-to{end_str}"

    filename = f"{symbol}-{output_suffix}{date_suffix}.pkl"

    # Determine output path
    if not output_path:
        if config and "resampling" in config and "output" in config["resampling"]:
            output_dir = config["resampling"]["output"].get("dir", "data/resampled")
            output_path = os.path.join(output_dir, filename)
        else:
            output_path = f"data/resampled/{filename}"

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save resampled data
    result.to_pickle(output_path)
    print(f"Resampled data saved to {output_path}")

    print("=" * 60)
    print("RESAMPLING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return result, output_path


@click.command()
@click.option("--config", type=str, help="Path to YAML config file")
@click.option("--input", type=str, help="Input file path")
@click.option("--output", type=str, help="Output file path")
@click.option(
    "--type",
    type=click.Choice(["time", "dollar"]),
    help="Resampling type: time or dollar bars",
)
@click.option(
    "--minutes", type=int, help="Minutes for time-based resampling (e.g., 1, 5, 15, 60)"
)
@click.option(
    "--threshold", type=float, help="Dollar threshold for dollar bars (e.g., 58000000)"
)
@click.option(
    "--start-date",
    type=str,
    help='Start date for filtering (YYYY-MM-DD or "YYYY-MM-DD HH:MM:SS")',
)
@click.option(
    "--end-date",
    type=str,
    help='End date for filtering (YYYY-MM-DD or "YYYY-MM-DD HH:MM:SS")',
)
@click.option(
    "--symbol",
    type=str,
    help="Trading symbol (e.g., USDJPY, EURUSD) for filename generation",
)
def main(
    config,
    input,
    output,
    type,
    minutes,
    threshold,
    start_date,
    end_date,
    symbol,
):
    """Forex Data Resampling Pipeline

    This script handles resampling of tick data to time-based or event-based bars.
    Supports both time-based OHLCV bars and dollar bars based on configuration.
    """
    # Load config if provided
    config_data = None
    if config:
        config_data = load_config(config)
        print(f"Loaded config from {config}")

    # Perform resampling
    resample_data(
        config=config_data,
        input_path=input,
        output_path=output,
        resample_type=type,
        minutes=minutes,
        threshold=threshold,
        start_date=start_date,
        end_date=end_date,
        symbol=symbol,
    )


if __name__ == "__main__":
    main()
