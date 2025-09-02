#!/usr/bin/env python3
"""
Generate direction labels for forex trading ML models using event-based sampling.

This script implements the triple-barrier labeling method from the notebook
Labelilng_1.event_based_direction_prediction.ipynb in a production-ready format.
"""

from pathlib import Path
from typing import Optional

import click
import pandas as pd

from libs.labeling import create_direction_labels
from libs.utils import load_config, build_file_paths_from_config


@click.command()
@click.option(
    "--config",
    "-c",
    default="config/config.yaml",
    help="Path to configuration YAML file",
)
@click.option("--symbol", "-s", help="Trading symbol (overrides config)")
@click.option(
    "--sample-type",
    type=click.Choice(["time", "dollar"]),
    help="Sampling type (overrides config)",
)
@click.option(
    "--minutes",
    "-m",
    type=int,
    help="Minutes for time-based sampling (overrides config)",
)
@click.option(
    "--dollar-threshold", help="Dollar threshold for dollar bars (overrides config)"
)
@click.option("--start-date", help="Start date YYYYMMDD (overrides config)")
@click.option("--end-date", help="End date YYYYMMDD (overrides config)")
@click.option(
    "--atr-window", type=int, help="ATR window for volatility (overrides config)"
)
@click.option(
    "--barrier-minutes", type=int, help="Vertical barrier minutes (overrides config)"
)
@click.option(
    "--pt-multiplier", type=float, help="Profit taking multiplier (overrides config)"
)
@click.option(
    "--sl-multiplier", type=float, help="Stop loss multiplier (overrides config)"
)
@click.option(
    "--min-ret-factor", type=float, help="Minimum return factor (overrides config)"
)
@click.option(
    "--num-threads",
    type=int,
    help="Number of threads for processing (overrides config)",
)
@click.option(
    "--no-intraday", is_flag=True, help="Include overnight events (overrides config)"
)
@click.option("--output-path", "-o", help="Custom output file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def generate_labels(
    config: str,
    symbol: Optional[str] = None,
    sample_type: Optional[str] = None,
    minutes: Optional[int] = None,
    dollar_threshold: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    atr_window: Optional[int] = None,
    barrier_minutes: Optional[int] = None,
    pt_multiplier: Optional[float] = None,
    sl_multiplier: Optional[float] = None,
    min_ret_factor: Optional[float] = None,
    num_threads: Optional[int] = None,
    no_intraday: bool = False,
    output_path: Optional[str] = None,
    verbose: bool = False,
):
    """
    Generate direction labels for forex trading ML models.

    This command processes OHLC data and event timestamps to create
    triple-barrier labels for machine learning model training.
    """


    try:
        # Load configuration
        if verbose:
            click.echo(f"Loading configuration from {config}")
        config_data = load_config(config)
        
        # Override config with CLI parameters if provided
        if symbol:
            config_data['resampling']['symbol'] = symbol
        if sample_type:
            config_data['resampling']['type'] = sample_type
        if minutes:
            config_data['resampling']['minutes'] = minutes
        if dollar_threshold:
            config_data['resampling']['threshold'] = dollar_threshold
        if start_date:
            config_data['resampling']['date_range']['start_date'] = start_date
        if end_date:
            config_data['resampling']['date_range']['end_date'] = end_date

        # Build file paths
        paths, sample_event, label_event = build_file_paths_from_config(config_data)
        
        # Handle event file naming convention mismatch
        # The existing files use format: {resampled_name}_{event_type}_{threshold:.2e}.pkl
        # while utils expects: {resampled_name}-{event_type}.pkl
        event_threshold = config_data['events']['threshold']
        event_type = config_data['events']['type']
        symbol = config_data['resampling']['symbol']
        
        if config_data['resampling']['type'] == 'dollar':
            threshold_val = config_data['resampling']['threshold']
            resampled_name = f"{symbol}-{threshold_val}-dollar-{config_data['resampling']['date_range']['start_date']}-{config_data['resampling']['date_range']['end_date']}"
        else:
            minutes = config_data['resampling']['minutes']
            start_date = config_data['resampling']['date_range']['start_date']
            end_date = config_data['resampling']['date_range']['end_date']
            resampled_name = f"{symbol}-{minutes}m-{start_date}-{end_date}"
        
        # Try alternative event file naming
        events_alt_path = Path(f"data/events/{resampled_name}_{event_type}_{event_threshold:.2e}.pkl")
        if events_alt_path.exists():
            paths['events'] = events_alt_path
        
        if verbose:
            click.echo(f"Processing: {symbol}-{sample_event}")
            click.echo(f"Event type: {label_event}")

        # Load data files
        if verbose:
            click.echo(f"Loading processed data from {paths['processed']}")
        if not paths['processed'].exists():
            raise FileNotFoundError(f"Processed data file not found: {paths['processed']}")
            
        df = pd.read_pickle(paths['processed'])
        if verbose:
            click.echo(f"Loaded OHLC data with {len(df)} rows")
        
        if verbose:
            click.echo(f"Loading events from {paths['events']}")
        if not paths['events'].exists():
            raise FileNotFoundError(f"Events file not found: {paths['events']}")
            
        t_events = pd.read_pickle(paths['events'])
        if verbose:
            click.echo(f"Loaded {len(t_events)} events")

        # Set parameters (CLI overrides config)
        labeling_config = config_data["labeling"]

        params = {
            "atr_window": atr_window or labeling_config["volatility"]["atr_window"],
            "barrier_delta_minutes": barrier_minutes
            or labeling_config["barriers"]["vertical_barrier_minutes"],
            "pt_sl_multipliers": (
                pt_multiplier or labeling_config["barriers"]["pt_sl_multipliers"][0],
                sl_multiplier or labeling_config["barriers"]["pt_sl_multipliers"][1],
            ),
            "min_ret_factor": min_ret_factor
            or labeling_config["barriers"]["min_ret_factor"],
            "num_threads": num_threads or labeling_config["processing"]["num_threads"],
            "intraday_only": not no_intraday
            and labeling_config["processing"]["intraday_only"],
        }

        if verbose:
            click.echo("Label generation parameters:")
            for key, value in params.items():
                click.echo(f"  {key}: {value}")

        # Generate labels
        if verbose:
            click.echo("Starting label generation...")
        labeled_events = create_direction_labels(df, t_events, **params)
        
        click.echo(f"Generated {len(labeled_events)} labeled events")
        label_counts = labeled_events['bin_class'].value_counts().sort_index()
        click.echo("Label distribution:")
        for label, count in label_counts.items():
            label_name = {0.0: "DOWN", 1.0: "NEUTRAL", 2.0: "UP"}[label]
            click.echo(f"  {label_name} ({label}): {count}")

        # Save results
        output_file = Path(output_path) if output_path else paths['direction_labels']
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            click.echo(f"Saving labeled events to {output_file}")
        labeled_events.to_pickle(output_file)
        
        # Save summary statistics
        summary_file = output_file.with_suffix('.summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Direction Labels Summary\n")
            f.write(f"=======================\n\n")
            f.write(f"Input files:\n")
            f.write(f"  Processed data: {paths['processed']}\n")
            f.write(f"  Events: {paths['events']}\n\n")
            f.write(f"Parameters:\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nResults:\n")
            f.write(f"  Total labeled events: {len(labeled_events)}\n")
            f.write(f"  Label distribution:\n")
            for label, count in label_counts.items():
                label_name = {0.0: "DOWN", 1.0: "NEUTRAL", 2.0: "UP"}[label]
                pct = count / len(labeled_events) * 100
                f.write(f"    {label_name} ({label}): {count} ({pct:.1f}%)\n")
            f.write(f"\nOutput file: {output_file}\n")
        
        if verbose:
            click.echo(f"Summary saved to {summary_file}")
        click.echo("Label generation completed successfully!")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.ClickException(str(e))


if __name__ == "__main__":
    generate_labels()

