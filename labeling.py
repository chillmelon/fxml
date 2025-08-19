#!/usr/bin/env python3
"""
Triple Barrier Labeling Script for Financial Time Series

This script implements the triple barrier labeling method from "Advances in Financial Machine Learning"
by Marcos López de Prado. It reads configuration from YAML files and applies the labeling method
to generate directional labels based on which barrier is hit first.

Usage:
    python labeling.py --config config/config.yaml --input data/processed/USDJPY-processed.pkl
    python labeling.py --help

Features:
- Configurable volatility methods (intraday, ATR, daily)
- CUSUM event detection
- Triple barrier method (profit taking, stop loss, time barriers)
- Parallel processing support
- Saves labeled events and classification bins
"""

import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import numpy as np
import pandas as pd
import yaml
from ta.volatility import AverageTrueRange
from tqdm import tqdm

from utils import build_file_paths_from_config

warnings.filterwarnings("ignore")

# Constants
DEFAULT_VOLATILITY_WINDOW = 60
DEFAULT_VOLATILITY_SPAN = 60
DEFAULT_TIME_BARRIER_HOURS = 3.0
DEFAULT_NUM_THREADS = 4
DEFAULT_DAILY_VOLATILITY_SPAN = 100
DEFAULT_VOLATILITY_MULTIPLIER = 1.0
DEFAULT_MIN_RETURN = 0.0


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if "labeling" not in config:
        raise KeyError("No 'labeling' section found in configuration file")

    return config


def get_tevents_optimized(data: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """Extract event timestamps using CUSUM filter."""
    values = data.values
    timestamps = data.index

    t_events_mask = np.zeros_like(values, dtype=bool)
    cum_pos, cum_neg = 0.0, 0.0

    for i in tqdm(range(len(values)), desc="Applying CUSUM filter"):
        cum_pos = max(0.0, cum_pos + values[i])
        cum_neg = min(0.0, cum_neg + values[i])

        if cum_pos > threshold:
            t_events_mask[i] = True
            cum_pos = 0.0
        if cum_neg < -threshold:
            t_events_mask[i] = True
            cum_neg = 0.0

    return timestamps[t_events_mask]


def get_vertical_barrier(
    t_events: pd.DatetimeIndex, close: pd.Series, delta: pd.Timedelta
) -> pd.Series:
    """Create vertical barriers (time limits) for each event."""
    barrier_times = t_events + delta
    t1_idx = close.index.searchsorted(barrier_times)
    valid_idx = t1_idx[t1_idx < len(close)]
    return pd.Series(close.index[valid_idx], index=t_events[: len(valid_idx)])


def get_intraday_vol(
    log_return: pd.Series, window: int = 60, span: int = 60
) -> pd.Series:
    """Calculate intraday volatility using rolling standard deviation and smoothing."""
    rolling_std = log_return.rolling(window=window).std()
    smoothed_vol = rolling_std.ewm(span=span).mean()
    return smoothed_vol.rename(f"intraday_vol_{window}_{span}")


def get_atr_volatility(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """Calculate ATR-based volatility."""
    atr = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=window
    )
    return atr.average_true_range().rename(f"atr{window}")


def get_daily_vol(close: pd.Series, span0: int = 100) -> pd.Series:
    """Calculate daily volatility using exponentially weighted moving average."""
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(
        close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0] :]
    )

    try:
        df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    except Exception:
        cut = close.loc[df0.index].shape[0] - close.loc[df0.values].shape[0]
        df0 = close.loc[df0.index].iloc[:-cut] / close.loc[df0.values].values - 1

    return df0.ewm(span=span0).std().rename("dailyVol")


def apply_pt_sl_on_t1(
    close: pd.Series, events: pd.DataFrame, pt_sl: List[float], molecule: pd.Index
) -> pd.DataFrame:
    """Apply profit taking and stop loss barriers for a subset of events."""
    events_ = events.loc[molecule]
    out = events_[["t1"]].copy(deep=True)

    pt = pt_sl[0] * events_["trgt"] if pt_sl[0] > 0 else pd.Series(index=events_.index)
    sl = -pt_sl[1] * events_["trgt"] if pt_sl[1] > 0 else pd.Series(index=events_.index)

    for loc, t1 in events_["t1"].fillna(close.index[-1]).items():
        df0 = close[loc:t1]
        df0 = (df0 / close[loc] - 1) * events_.at[loc, "side"]
        out.loc[loc, "sl"] = df0[df0 < sl[loc]].index.min()
        out.loc[loc, "pt"] = df0[df0 > pt[loc]].index.min()

    return out


def parallel_apply(
    func, items: pd.Index, num_threads: int = 4, **kwargs
) -> pd.DataFrame:
    """Apply a function in parallel across chunks of items."""

    def worker(molecule):
        return func(molecule=molecule, **kwargs)

    chunks = np.array_split(items, num_threads)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(worker, chunks))

    return pd.concat(results).sort_index()


def get_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: List[float],
    trgt: pd.Series,
    min_ret: float = 0.0,
    num_threads: int = 4,
    t1: Union[bool, pd.Series] = False,
    side: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Apply triple barrier method to generate labeled events."""
    # Step 1: Filter targets
    trgt = trgt.loc[t_events]
    trgt = trgt[trgt > min_ret]

    # Step 2: Set vertical barrier (t1)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=t_events)

    # Step 3: Build events DataFrame
    if side is None:
        side_, pt_sl_ = pd.Series(1.0, index=trgt.index), [pt_sl[0], pt_sl[0]]
    else:
        side_, pt_sl_ = side.loc[trgt.index], pt_sl[:2]

    events = pd.concat({"t1": t1, "trgt": trgt, "side": side_}, axis=1).dropna(
        subset=["trgt"]
    )

    # Step 4: Apply barriers in parallel
    df0 = parallel_apply(
        func=apply_pt_sl_on_t1,
        items=events.index,
        num_threads=num_threads,
        close=close,
        events=events,
        pt_sl=pt_sl_,
    )

    # Step 5: Choose the first touched barrier
    events["t1"] = df0.dropna(how="all").min(axis=1)

    if side is None:
        events = events.drop("side", axis=1)

    return events


def get_bins(
    events: pd.DataFrame, close: pd.Series, t1: Optional[pd.Series] = None
) -> pd.DataFrame:
    """Create classification labels from triple barrier events."""
    # 1) Prices aligned with events
    events_ = events.dropna(subset=["t1"])
    px = events_.index.union(events_["t1"].values).drop_duplicates()
    px = close.reindex(px, method="bfill")

    # 2) Create output object
    out = pd.DataFrame(index=events_.index)
    out["ret"] = px.loc[events_["t1"].values].values / px.loc[events_.index] - 1

    if "side" in events_:
        out["ret"] *= events_["side"]

    out["bin"] = np.sign(out["ret"])

    if "side" not in events_:
        if t1 is not None:
            vtouch_first_idx = events[events["t1"].isin(t1.values)].index
            out.loc[vtouch_first_idx, "bin"] = 0.0

    if "side" in events_:
        out.loc[out["ret"] <= 0, "bin"] = 0

    return out


def get_concurrency(events: pd.DataFrame, price_index: pd.DatetimeIndex) -> pd.Series:
    """Calculate label concurrency for sample weighting."""
    concurrency = pd.Series(0, index=price_index)

    for start, end in events["t1"].items():
        if pd.notna(end):
            concurrency[start:end] += 1

    return concurrency


def get_target_volatility(
    df: pd.DataFrame, log_returns: pd.Series, config: Dict[str, Any]
) -> pd.Series:
    """Calculate target volatility based on configuration settings."""
    method = config.get("volatility_method", "intraday")
    window = config.get("volatility_window", 60)
    span = config.get("volatility_span", 60)

    if method == "atr":
        return get_atr_volatility(df, window=window)
    elif method == "intraday":
        return get_intraday_vol(log_returns, window=window, span=span)
    elif method == "daily":
        return get_daily_vol(df["close"], span0=span)
    else:
        raise ValueError(f"Unknown volatility method: {method}")


def apply_triple_barrier_labeling(
    df: pd.DataFrame, labeling_config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply triple barrier labeling method using configuration."""

    close = df["close"]
    log_returns = df["close_log_return"]

    # Get configuration parameters
    vol_multiplier = labeling_config.get("vol_multiplier", 1.0)
    pt_sl = labeling_config.get("pt_sl", [1.0, 1.0])
    time_barrier_hours = labeling_config.get("time_barrier_hours", 3.0)
    min_ret = labeling_config.get("min_ret", 0.0)
    num_threads = labeling_config.get("num_threads", 4)
    intraday_only = labeling_config.get("intraday_only", True)

    click.echo(f"Configuration:")
    click.echo(
        f"  - Volatility method: {labeling_config.get('volatility_method', 'intraday')}"
    )
    click.echo(f"  - Volatility multiplier: {vol_multiplier}")
    click.echo(f"  - PT/SL barriers: {pt_sl}")
    click.echo(f"  - Time barrier: {time_barrier_hours} hours")
    click.echo(f"  - Minimum return: {min_ret}")
    click.echo(f"  - Threads: {num_threads}")

    # Step 1: Calculate target volatility
    click.echo("Step 1: Calculating target volatility...")
    trgt_vol = get_target_volatility(df, log_returns, labeling_config)

    # Step 2: Calculate CUSUM threshold
    click.echo("Step 2: Calculating CUSUM threshold...")
    if labeling_config.get("volatility_method") == "atr":
        threshold = trgt_vol.median() * vol_multiplier
    else:
        vol = log_returns.rolling(
            window=labeling_config.get("volatility_window", 60)
        ).std()
        threshold = vol.mean() * vol_multiplier

    click.echo(f"Step 3: Extracting events with threshold: {threshold:.6f}")
    t_events = get_tevents_optimized(log_returns.iloc[1:], threshold=threshold)

    click.echo("Step 4: Creating vertical barriers...")
    time_horizon = pd.Timedelta(hours=time_barrier_hours)
    t1 = get_vertical_barrier(t_events, close, delta=time_horizon)

    click.echo("Step 5: Reindexing target volatility to events...")
    trgt = trgt_vol.reindex(t_events, method="ffill")

    click.echo("Step 6: Applying triple barrier method...")
    events = get_events(
        close=close,
        t_events=t_events,
        pt_sl=pt_sl,
        trgt=trgt,
        min_ret=min_ret,
        num_threads=num_threads,
        t1=t1,
        side=None,
    )

    click.echo("Step 7: Creating classification labels...")
    labels = get_bins(events, close, t1=t1)

    # Step 8: Optional intraday filtering
    if intraday_only:
        click.echo("Step 8: Filtering intraday events...")
        same_day_mask = events.index.date == events["t1"].dt.date
        events = events[same_day_mask]
        labels = labels[same_day_mask]

    # Join events and labels
    labeled_events = events.join(labels, how="inner")
    labeled_events["bin_class"] = labeled_events["bin"] + 1

    click.echo(f"\nGenerated {len(events)} labeled events")
    click.echo("Label distribution:")
    label_counts = labels["bin"].value_counts().sort_index()
    for label, count in label_counts.items():
        click.echo(f"  {label}: {count}")

    return events, labeled_events


@click.command()
@click.option(
    "--config",
    "-c",
    default="config/config.yaml",
    help="Path to configuration YAML file",
)
@click.option(
    "--input",
    "-i",
    help="Path to input processed data file (pickle)",
    default=None,
)
@click.option(
    "--output-dir",
    "-o",
    default=None,
    help="Output directory for labeled data (default: from config)",
)
@click.option(
    "--event-name",
    "-e",
    default="cusum_filter",
    help="Name for the labeling event type",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def main(config: str, input: str, output_dir: str, event_name: str, verbose: bool):
    """
    Apply triple barrier labeling to financial time series data.

    This script reads processed price data and applies the triple barrier method
    to generate directional labels based on profit taking, stop loss, and time barriers.

    Example:
        python labeling.py -i data/processed/USDJPY-processed.pkl -c config/config.yaml
    """

    if verbose:
        click.echo(f"Loading configuration from: {config}")

    # Load configuration
    try:
        full_config = load_config(config)
        labeling_config = full_config["labeling"]
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        return 1

    # Set output directory
    if output_dir is None:
        output_dir = labeling_config.get("output", {}).get(
            "dir", "data/direction_labels"
        )

    if input is None:
        paths, _, _ = build_file_paths_from_config(full_config)
        input = str(paths["processed"])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        click.echo(f"Output directory: {output_path}")

    # Load input data
    input_path = Path(input)
    if not input_path.exists():
        click.echo(f"Input file not found: {input}", err=True)
        return 1

    click.echo(f"Loading data from: {input_path}")

    try:
        df = pd.read_pickle(input_path)
        click.echo(f"Loaded data shape: {df.shape}")

        # Check required columns
        required_cols = ["close", "close_log_return"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            click.echo(f"Missing required columns: {missing_cols}", err=True)
            return 1

        # Check if OHLC data is available for ATR
        if labeling_config.get("volatility_method") == "atr":
            ohlc_cols = ["high", "low", "close"]
            missing_ohlc = [col for col in ohlc_cols if col not in df.columns]
            if missing_ohlc:
                click.echo(
                    f"ATR method requires OHLC data. Missing: {missing_ohlc}", err=True
                )
                return 1

    except Exception as e:
        click.echo(f"Error loading data: {e}", err=True)
        return 1

    # Apply triple barrier labeling
    try:
        events, labeled_events = apply_triple_barrier_labeling(df, labeling_config)
    except Exception as e:
        click.echo(f"Error applying labeling: {e}", err=True)
        return 1

    # Generate output filename
    input_stem = input_path.stem
    output_filename = f"{input_stem}-{event_name}.pkl"
    output_file = output_path / output_filename

    # Save results
    click.echo(f"Saving labeled events to: {output_file}")
    try:
        labeled_events.to_pickle(output_file)

        click.echo(f"✅ Successfully saved labeled events:")
        click.echo(f"   - Labeled events: {output_file}")

        # Print summary statistics
        click.echo(f"\n📊 Summary:")
        click.echo(f"   - Total events: {len(labeled_events)}")
        click.echo(
            f"   - Date range: {labeled_events.index.min()} to {labeled_events.index.max()}"
        )

        # Calculate and show concurrency
        if verbose:
            click.echo("Calculating label concurrency...")
            concurrency = get_concurrency(events, df.index)
            click.echo(f"   - Average concurrency: {concurrency.mean():.2f}")
            click.echo(f"   - Max concurrency: {concurrency.max()}")

    except Exception as e:
        click.echo(f"Error saving results: {e}", err=True)
        return 1

    click.echo("🎉 Labeling completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
