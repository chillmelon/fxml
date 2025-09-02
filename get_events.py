#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Optional, Tuple

import click
import pandas as pd

from libs.events import EventProcessor, validate_event_data
from libs.utils import build_file_paths_from_config, load_config


def load_and_validate_data(
    input_path: Path, column: str, verbose: bool = False
) -> pd.Series:
    """Load data file and validate the specified column."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist")

    if verbose:
        click.echo(f"Loading data from {input_path}")

    try:
        df = pd.read_pickle(input_path)
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")

    if verbose:
        click.echo(f"Loaded data with shape {df.shape}")
        click.echo(f"Using column: {column}")

    # Use validation function from libs.events
    return validate_event_data(df, column)


def resolve_configuration(
    config_path: Optional[str],
    input_file: Optional[str],
    threshold: Optional[float],
    verbose: bool = False,
) -> Tuple[Path, float, str]:
    """Resolve input path, threshold, and event name from config and CLI args."""
    config_data = None
    if config_path:
        config_data = load_config(config_path)
        if verbose:
            click.echo(f"Loaded config from {config_path}")

    # Determine input file path
    if input_file:
        input_path = Path(input_file)
    elif config_data:
        paths, _, _ = build_file_paths_from_config(config_data)
        input_path = paths["processed"]
        if verbose:
            click.echo(f"Using processed file from config: {input_path}")
    else:
        raise ValueError("Either --input-file or --config must be provided")

    # Determine threshold
    if threshold is None:
        if (
            config_data
            and "events" in config_data
            and "threshold" in config_data["events"]
        ):
            threshold = config_data["events"]["threshold"]
            if verbose:
                click.echo(f"Using threshold from config: {threshold:.6f}")
        else:
            raise ValueError("Threshold not specified in config or CLI")
    else:
        if verbose:
            click.echo(f"Using provided threshold: {threshold:.6f}")

    # Determine event name
    event_name = "CUSUM"  # default
    if config_data and "events" in config_data and "type" in config_data["events"]:
        event_name = config_data["events"]["type"].upper()
        if verbose:
            click.echo(f"Using event name from config: {event_name}")

    return input_path, threshold, event_name


def generate_output_path(
    config_path: Optional[str],
    input_path: Path,
    output_dir: str,
    threshold: float,
    event_name: str,
) -> Path:
    """Generate consistent output file path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if config_path:
        config_data = load_config(config_path)
        paths, _, _ = build_file_paths_from_config(config_data)
        return paths["events"]
    else:
        # Generate filename from input path
        input_basename = input_path.stem.replace("-processed", "")
        filename = f"{input_basename}_{event_name}_{threshold:.2e}.pkl"
        return output_path / filename


def save_events(
    events: pd.DatetimeIndex, output_path: Path, verbose: bool = False
) -> None:
    """Save events to file with error handling."""
    try:
        events.to_series().to_pickle(output_path)
        click.echo(f"Events saved to: {output_path}")

        if verbose and len(events) > 0:
            click.echo(f"Event date range: {events[0]} to {events[-1]}")
            avg_time = pd.Series(events).diff().mean()
            click.echo(f"Average time between events: {avg_time}")

    except Exception as e:
        raise RuntimeError(f"Error saving events: {e}")


@click.command()
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Path to YAML config file"
)
@click.option(
    "--input-file",
    "-i",
    help="Path to input processed data file (.pkl). If not provided, will use config",
)
@click.option(
    "--output-dir",
    "-o",
    default="data/events",
    help="Output directory for events file (default: data/events)",
)
@click.option(
    "--column",
    "--col",
    default="close_log_return",
    help="Column to use for event detection (default: close_log_return)",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    help="CUSUM threshold.",
)
@click.option(
    "--algorithm",
    "-a",
    default="cusum",
    help="Event detection algorithm (default: cusum)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def main(
    config,
    input_file,
    output_dir,
    column,
    threshold,
    algorithm,
    verbose,
):
    """
    Extract events using various algorithms from processed market data.

    This script implements event detection algorithms for financial time series data.
    The default CUSUM filter detects events when cumulative positive or negative
    movements exceed a specified threshold.

    Example usage:
        python get_events.py --config config/config.yaml
        python get_events.py -c config/config.yaml --verbose
        python get_events.py -i data/processed/USDJPY-1m-20210101-20241231-processed.pkl
        python get_events.py -i data/processed/data.pkl -t 0.001 --column close_log_return
    """
    try:
        # Resolve configuration and parameters
        input_path, resolved_threshold, event_name = resolve_configuration(
            config, input_file, threshold, verbose
        )

        # Load and validate data
        data_series = load_and_validate_data(input_path, column, verbose)

        # Skip first row to avoid NaN issues
        data_series = data_series.iloc[1:]

        if verbose:
            click.echo(f"Processing {len(data_series)} data points")
            click.echo(f"Using algorithm: {algorithm}")

        # Initialize event processor and extract events
        processor = EventProcessor(algorithm=algorithm)
        events = processor.extract_events(data_series, threshold=resolved_threshold)

        click.echo(f"Found {len(events)} events")

        # Generate output path and save
        output_file_path = generate_output_path(
            config, input_path, output_dir, resolved_threshold, event_name
        )
        save_events(events, output_file_path, verbose)

    except (ValueError, FileNotFoundError, RuntimeError) as e:
        click.echo(f"Error: {e}", err=True)
        return
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        return


if __name__ == "__main__":
    main()
