import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta


def macd(
    close: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
):
    # Convert to pandas Series for pandas_ta
    close_series = pd.Series(close)

    # Calculate MACD using pandas_ta
    macd_df = ta.macd(
        close_series, fast=fast_period, slow=slow_period, signal=signal_period
    )

    # Extract components
    macd_line = macd_df[f"MACD_{fast_period}_{slow_period}_{signal_period}"].to_numpy()
    signal_line = macd_df[
        f"MACDs_{fast_period}_{slow_period}_{signal_period}"
    ].to_numpy()
    histogram = macd_df[f"MACDh_{fast_period}_{slow_period}_{signal_period}"].to_numpy()

    return macd_line, signal_line, histogram


if __name__ == "__main__":
    # Load data
    data = pd.read_pickle("data/resampled/EURUSD-60m-20240101-20241231.pkl")
    data["timestamp"] = data["timestamp"].astype("datetime64[s]")
    data = data.set_index("timestamp")
    data = data.dropna()

    # Calculate MACD
    macd_line, signal_line, histogram = macd(
        data["close"].to_numpy(), fast_period=12, slow_period=26, signal_period=9
    )

    data["macd"] = macd_line
    data["signal"] = signal_line
    data["histogram"] = histogram

    # Visualize
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Price plot
    data["close"].plot(ax=ax1, label="Close", color="white")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.set_title("EUR/USD Price and MACD Indicator")

    # MACD plot
    data["macd"].plot(ax=ax2, label="MACD", color="blue")
    data["signal"].plot(ax=ax2, label="Signal", color="red")
    ax2.fill_between(
        data.index, data["histogram"], 0, alpha=0.3, color="yellow", label="Histogram"
    )

    ax2.axhline(0, color="white", linestyle="--", alpha=0.5)
    ax2.set_ylabel("MACD")
    ax2.legend()

    plt.tight_layout()
    plt.show()
