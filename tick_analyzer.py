#!/usr/bin/env python
# coding: utf-8

"""
Tick Data Analyzer

This script analyzes raw tick data to provide insights and recommendations for resampling.
It examines data quality, patterns, and generates optimal resampling parameters.

Usage:
    python tick_analyzer.py --input data/raw/tick_data.csv
    python tick_analyzer.py --input data/raw/tick_data.csv --sample-size 100000
    python tick_analyzer.py --input data/raw/tick_data.csv --output analysis_report.json
"""

import json
import os
from datetime import datetime

import click
import pandas as pd


def load_tick_data(file_path):
    """Load tick data from CSV or pickle file with optional sampling"""
    print(f"Loading tick data from {file_path}")

    if file_path.endswith(".pkl"):
        df = pd.read_pickle(file_path)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .pkl or .csv")

    print(f"Loaded {len(df):,} tick records")
    return df


def analyze_data_quality(df):
    """Analyze data quality metrics"""
    print("\n" + "=" * 50)
    print("DATA QUALITY ANALYSIS")
    print("=" * 50)

    quality_report = {}

    # Basic statistics
    quality_report["total_records"] = len(df)
    quality_report["columns"] = list(df.columns)

    # Missing values
    missing_data = df.isnull().sum()
    quality_report["missing_values"] = missing_data.to_dict()

    print(f"Total records: {quality_report['total_records']:,}")
    print(f"Columns: {quality_report['columns']}")

    if missing_data.sum() > 0:
        print("\nMissing values:")
        for col, count in missing_data.items():
            if count > 0:
                pct = (count / len(df)) * 100
                print(f"  {col}: {count:,} ({pct:.2f}%)")
    else:
        print("No missing values found")

    # Convert timestamp if needed
    if "timestamp" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Time range analysis
    if "timestamp" in df.columns:
        start_time = df["timestamp"].min()
        end_time = df["timestamp"].max()
        duration = end_time - start_time

        quality_report["time_range"] = {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "duration_days": duration.days,
            "duration_hours": duration.total_seconds() / 3600,
        }

        print(f"\nTime range:")
        print(f"  Start: {start_time}")
        print(f"  End: {end_time}")
        print(
            f"  Duration: {duration.days} days ({duration.total_seconds()/3600:.1f} hours)"
        )

    # Price analysis
    if "askPrice" in df.columns and "bidPrice" in df.columns:
        df["mid"] = (df["askPrice"] + df["bidPrice"]) / 2
        df["spread"] = df["askPrice"] - df["bidPrice"]

        price_stats = {
            "mid_price": {
                "min": float(df["mid"].min()),
                "max": float(df["mid"].max()),
                "mean": float(df["mid"].mean()),
                "std": float(df["mid"].std()),
            },
            "spread": {
                "min": float(df["spread"].min()),
                "max": float(df["spread"].max()),
                "mean": float(df["spread"].mean()),
                "std": float(df["spread"].std()),
            },
        }
        quality_report["price_analysis"] = price_stats

        print(f"\nPrice analysis:")
        print(
            f"  Mid price range: {price_stats['mid_price']['min']:.5f} - {price_stats['mid_price']['max']:.5f}"
        )
        print(
            f"  Average spread: {price_stats['spread']['mean']:.5f} ± {price_stats['spread']['std']:.5f}"
        )

    # Volume analysis
    if "askVolume" in df.columns and "bidVolume" in df.columns:
        # Volume is in millions, convert to actual volume
        df["total_volume"] = (df["askVolume"] + df["bidVolume"]) * 1_000_000

        volume_stats = {
            "total_volume": {
                "min": float(df["total_volume"].min()),
                "max": float(df["total_volume"].max()),
                "mean": float(df["total_volume"].mean()),
                "std": float(df["total_volume"].std()),
            }
        }
        quality_report["volume_analysis"] = volume_stats

        print(f"\nVolume analysis:")
        print(
            f"  Volume range: {volume_stats['total_volume']['min']:.2f} - {volume_stats['total_volume']['max']:.2f}"
        )
        print(
            f"  Average volume: {volume_stats['total_volume']['mean']:.2f} ± {volume_stats['total_volume']['std']:.2f}"
        )

    return quality_report


def analyze_tick_patterns(df):
    """Analyze tick arrival patterns and frequency"""
    print("\n" + "=" * 50)
    print("TICK PATTERN ANALYSIS")
    print("=" * 50)

    pattern_report = {}

    if "timestamp" not in df.columns:
        print("No timestamp column found - skipping pattern analysis")
        return pattern_report

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Calculate tick intervals
    df_sorted = df.sort_values("timestamp")
    tick_intervals = df_sorted["timestamp"].diff().dt.total_seconds()
    tick_intervals = tick_intervals.dropna()

    interval_stats = {
        "mean_seconds": float(tick_intervals.mean()),
        "median_seconds": float(tick_intervals.median()),
        "std_seconds": float(tick_intervals.std()),
        "min_seconds": float(tick_intervals.min()),
        "max_seconds": float(tick_intervals.max()),
    }
    pattern_report["tick_intervals"] = interval_stats

    print(f"Tick interval statistics (seconds):")
    print(f"  Mean: {interval_stats['mean_seconds']:.3f}")
    print(f"  Median: {interval_stats['median_seconds']:.3f}")
    print(f"  Std: {interval_stats['std_seconds']:.3f}")
    print(
        f"  Range: {interval_stats['min_seconds']:.3f} - {interval_stats['max_seconds']:.3f}"
    )

    # Ticks per time period
    df_sorted["hour"] = df_sorted["timestamp"].dt.hour
    df_sorted["day_of_week"] = df_sorted["timestamp"].dt.dayofweek

    hourly_counts = df_sorted.groupby("hour").size()
    daily_counts = df_sorted.groupby("day_of_week").size()

    pattern_report["hourly_distribution"] = hourly_counts.to_dict()
    pattern_report["daily_distribution"] = daily_counts.to_dict()

    print(f"\nActivity patterns:")
    print(
        f"  Most active hour: {hourly_counts.idxmax()} ({hourly_counts.max():,} ticks)"
    )
    print(
        f"  Least active hour: {hourly_counts.idxmin()} ({hourly_counts.min():,} ticks)"
    )
    print(f"  Most active day: {daily_counts.idxmax()} ({daily_counts.max():,} ticks)")
    print(f"  Least active day: {daily_counts.idxmin()} ({daily_counts.min():,} ticks)")

    return pattern_report


def generate_resampling_recommendations(df, quality_report, pattern_report):
    """Generate recommendations for optimal resampling parameters"""
    print("\n" + "=" * 50)
    print("RESAMPLING RECOMMENDATIONS")
    print("=" * 50)

    recommendations = {}

    # Time-based recommendations
    if "tick_intervals" in pattern_report:
        mean_interval = pattern_report["tick_intervals"]["mean_seconds"]
        median_interval = pattern_report["tick_intervals"]["median_seconds"]

        # Recommend time bars based on typical tick frequency
        recommended_minutes = []
        if mean_interval < 1:  # Very frequent ticks
            recommended_minutes = [1, 5, 15, 60]
        elif mean_interval < 10:  # Moderately frequent
            recommended_minutes = [5, 15, 60, 240]
        else:  # Less frequent
            recommended_minutes = [15, 60, 240, 1440]

        recommendations["time_bars"] = {
            "recommended_minutes": recommended_minutes,
            "reasoning": f"Based on mean tick interval of {mean_interval:.3f} seconds",
        }

        print("Time-based bar recommendations:")
        for minutes in recommended_minutes:
            if minutes < 60:
                print(f"  {minutes} minute bars")
            elif minutes < 1440:
                print(f"  {minutes//60} hour bars")
            else:
                print(f"  {minutes//1440} day bars")

    # Dollar bar recommendations
    if "price_analysis" in quality_report and "volume_analysis" in quality_report:
        avg_price = quality_report["price_analysis"]["mid_price"]["mean"]
        avg_volume_millions = (
            quality_report["volume_analysis"]["total_volume"]["mean"] / 1_000_000
        )
        avg_volume = avg_volume_millions * 1_000_000

        # Calculate average dollar value per tick
        avg_dollar_per_tick = avg_price * avg_volume

        # Recommend dollar thresholds based on typical trade sizes
        total_records = quality_report["total_records"]

        # Target different bar frequencies
        target_bars_per_day = [50, 100, 200, 500]  # Different granularities
        days = quality_report.get("time_range", {}).get("duration_days", 1)

        recommended_thresholds = []
        for bars_per_day in target_bars_per_day:
            target_total_bars = bars_per_day * days
            ticks_per_bar = total_records / target_total_bars
            threshold = avg_dollar_per_tick * ticks_per_bar
            recommended_thresholds.append(int(threshold))

        recommendations["dollar_bars"] = {
            "recommended_thresholds": recommended_thresholds,
            "avg_dollar_per_tick": avg_dollar_per_tick,
            "reasoning": f"Based on average ${avg_dollar_per_tick:.2f} per tick",
        }

        print("\nDollar bar recommendations:")
        for i, threshold in enumerate(recommended_thresholds):
            bars_per_day = target_bars_per_day[i]
            if threshold >= 1000000:
                threshold_str = f"${threshold//1000000}M"
            elif threshold >= 1000:
                threshold_str = f"${threshold//1000}K"
            else:
                threshold_str = f"${threshold}"
            print(f"  {threshold_str} ({bars_per_day} bars/day target)")

    # Data quality recommendations
    data_quality_score = 100
    issues = []

    if quality_report.get("missing_values", {}).get("askPrice", 0) > 0:
        data_quality_score -= 20
        issues.append("Missing price data")

    if quality_report.get("missing_values", {}).get("timestamp", 0) > 0:
        data_quality_score -= 30
        issues.append("Missing timestamp data")

    recommendations["data_quality"] = {
        "score": data_quality_score,
        "issues": issues,
        "preprocessing_needed": len(issues) > 0,
    }

    print(f"\nData quality score: {data_quality_score}/100")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Data quality is good - no preprocessing required")

    return recommendations


@click.command()
@click.option(
    "--input",
    "-i",
    required=True,
    type=str,
    help="Input tick data file path (.csv or .pkl)",
)
@click.option(
    "--output", "-o", type=str, help="Output analysis report file path (.json)"
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def main(input, output, verbose):
    """Analyze raw tick data and generate resampling recommendations.

    This tool examines tick data quality, patterns, and provides optimal
    resampling parameters for both time-based and dollar bars.
    """

    print("TICK DATA ANALYZER")
    print("=" * 60)

    # Load data
    try:
        df = load_tick_data(input)
    except Exception as e:
        click.echo(f"Error loading data: {e}", err=True)
        return

    # Perform analysis
    quality_report = analyze_data_quality(df)
    pattern_report = analyze_tick_patterns(df)
    recommendations = generate_resampling_recommendations(
        df, quality_report, pattern_report
    )

    # Compile full report
    full_report = {
        "metadata": {
            "analysis_timestamp": datetime.now().isoformat(),
            "input_file": input,
        },
        "data_quality": quality_report,
        "tick_patterns": pattern_report,
        "recommendations": recommendations,
    }

    # Output results
    if output:
        os.makedirs(
            os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True
        )
        with open(output, "w") as f:
            json.dump(full_report, f, indent=2, default=str)
        print(f"\nAnalysis report saved to: {output}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

