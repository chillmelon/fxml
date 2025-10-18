import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fxml.trading.strategies.macd_cross.macd_indicator import (
    calculate_macd,
    detect_crossover,
)


def macd_cross_strategy(
    close: np.array, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
):
    """
    MACD crossover trading strategy.

    Parameters
    ----------
    close : np.array
        Close price array
    fast_period : int
        Fast EMA period (default 12)
    slow_period : int
        Slow EMA period (default 26)
    signal_period : int
        Signal line EMA period (default 9)

    Returns
    -------
    tuple
        (macd_line, signal_line, histogram, trading_signals)
    """
    # Calculate MACD components
    macd_line, signal_line, histogram = calculate_macd(
        close, fast_period, slow_period, signal_period
    )

    # Generate trading signals
    sig = np.zeros(len(close))

    for i in range(1, len(close)):
        # Skip if NaN values
        if np.isnan(macd_line[i]) or np.isnan(signal_line[i]):
            sig[i] = 0.0
            continue

        # Bullish crossover: MACD crosses above signal
        if macd_line[i - 1] <= signal_line[i - 1] and macd_line[i] > signal_line[i]:
            sig[i] = 1.0
        # Bearish crossover: MACD crosses below signal
        elif macd_line[i - 1] >= signal_line[i - 1] and macd_line[i] < signal_line[i]:
            sig[i] = -1.0
        else:
            # Hold previous position
            sig[i] = sig[i - 1]

    return macd_line, signal_line, histogram, sig


def main():
    data = pd.read_pickle("data/resampled/EURUSD-60m-20240101-20241231.pkl")
    data["timestamp"] = data["timestamp"].astype("datetime64[s]")
    data = data.set_index("timestamp")
    data = data.dropna()

    # Default MACD parameters
    fast_period = 12
    slow_period = 26
    signal_period = 9

    macd_line, signal_line, histogram, signal = macd_cross_strategy(
        data["close"].to_numpy(), fast_period, slow_period, signal_period
    )
    data["macd"] = macd_line
    data["signal_line"] = signal_line
    data["histogram"] = histogram
    data["signal"] = signal

    # Visualize price and MACD
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    data["close"].plot(ax=ax1, label="Close", color="white")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.set_title(f"MACD Strategy ({fast_period}, {slow_period}, {signal_period})")

    data["macd"].plot(ax=ax2, label="MACD", color="blue")
    data["signal_line"].plot(ax=ax2, label="Signal", color="orange")
    ax2.fill_between(
        data.index, data["histogram"], 0, alpha=0.3, color="gray", label="Histogram"
    )
    ax2.axhline(0, color="white", linestyle="--", alpha=0.5)
    ax2.set_ylabel("MACD")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Calculate returns
    data["r"] = np.log(data["close"]).diff().shift(-1)
    strat_r = data["signal"] * data["r"]

    # Calculate profit factor
    pf = strat_r[strat_r > 0].sum() / strat_r[strat_r < 0].abs().sum()
    print(
        f"Profit Factor ({fast_period}, {slow_period}, {signal_period}): {pf:.4f}"
    )

    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    strat_r.cumsum().plot(label="Strategy", color="cyan")
    data["r"].cumsum().plot(label="Buy & Hold", color="gray", alpha=0.5)
    plt.ylabel("Cumulative Log Return")
    plt.title("MACD Strategy Performance")
    plt.legend()
    plt.show()

    ## Parameter Sweep - Fast Period
    print("\n=== Parameter Sweep: Fast Period ===")
    fast_periods = list(range(6, 20, 2))
    pfs = []

    for fp in fast_periods:
        macd_line, signal_line, histogram, signal = macd_cross_strategy(
            data["close"].to_numpy(), fp, slow_period, signal_period
        )
        data["signal"] = signal

        data["r"] = np.log(data["close"]).diff().shift(-1)
        strat_r = data["signal"] * data["r"]

        pf = strat_r[strat_r > 0].sum() / strat_r[strat_r < 0].abs().sum()
        print(f"Profit Factor (fast={fp}): {pf:.4f}")
        pfs.append(pf)

    plt.style.use("dark_background")
    plt.figure(figsize=(10, 6))
    x = pd.Series(pfs, index=fast_periods)
    x.plot(marker="o")
    plt.ylabel("Profit Factor")
    plt.xlabel("Fast Period")
    plt.title("MACD Strategy - Fast Period Optimization")
    plt.axhline(1.0, color="white", linestyle="--", alpha=0.5)
    plt.grid(alpha=0.3)
    plt.show()

    ## Parameter Sweep - Slow Period
    print("\n=== Parameter Sweep: Slow Period ===")
    slow_periods = list(range(20, 40, 2))
    pfs = []

    for sp in slow_periods:
        macd_line, signal_line, histogram, signal = macd_cross_strategy(
            data["close"].to_numpy(), fast_period, sp, signal_period
        )
        data["signal"] = signal

        data["r"] = np.log(data["close"]).diff().shift(-1)
        strat_r = data["signal"] * data["r"]

        pf = strat_r[strat_r > 0].sum() / strat_r[strat_r < 0].abs().sum()
        print(f"Profit Factor (slow={sp}): {pf:.4f}")
        pfs.append(pf)

    plt.figure(figsize=(10, 6))
    x = pd.Series(pfs, index=slow_periods)
    x.plot(marker="o", color="orange")
    plt.ylabel("Profit Factor")
    plt.xlabel("Slow Period")
    plt.title("MACD Strategy - Slow Period Optimization")
    plt.axhline(1.0, color="white", linestyle="--", alpha=0.5)
    plt.grid(alpha=0.3)
    plt.show()

    ## Parameter Sweep - Signal Period
    print("\n=== Parameter Sweep: Signal Period ===")
    signal_periods = list(range(5, 16, 1))
    pfs = []

    for sigp in signal_periods:
        macd_line, signal_line, histogram, signal = macd_cross_strategy(
            data["close"].to_numpy(), fast_period, slow_period, sigp
        )
        data["signal"] = signal

        data["r"] = np.log(data["close"]).diff().shift(-1)
        strat_r = data["signal"] * data["r"]

        pf = strat_r[strat_r > 0].sum() / strat_r[strat_r < 0].abs().sum()
        print(f"Profit Factor (signal={sigp}): {pf:.4f}")
        pfs.append(pf)

    plt.figure(figsize=(10, 6))
    x = pd.Series(pfs, index=signal_periods)
    x.plot(marker="o", color="green")
    plt.ylabel("Profit Factor")
    plt.xlabel("Signal Period")
    plt.title("MACD Strategy - Signal Period Optimization")
    plt.axhline(1.0, color="white", linestyle="--", alpha=0.5)
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
