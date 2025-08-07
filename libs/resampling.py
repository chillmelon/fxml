import pandas as pd


def create_dollar_bar(data: pd.DataFrame, dollar: float) -> pd.DataFrame:
    """
    Convert tick data into dollar bars.

    Parameters:
    - data: pd.DataFrame with columns:
        - timestamp: unix (ms)
        - askPrice, bidPrice
        - askVolume, bidVolume
    - dollar: float, target dollar value per bar

    Returns:
    - pd.DataFrame with columns:
        - timestamp, open, high, low, close, volume
    """

    bars = []
    cum_dollar = 0
    bar_prices = []
    bar_volumes = []
    bar_timestamps = []

    for i, row in data.iterrows():
        mid_price = (row["askPrice"] + row["bidPrice"]) / 2
        volume = min(row["askVolume"], row["bidVolume"])  # crude approximation
        trade_dollar = mid_price * volume

        cum_dollar += trade_dollar
        bar_prices.append(mid_price)
        bar_volumes.append(volume)
        bar_timestamps.append(row["timestamp"])

        if cum_dollar >= dollar:
            open_price = bar_prices[0]
            high_price = max(bar_prices)
            low_price = min(bar_prices)
            close_price = bar_prices[-1]
            total_volume = sum(bar_volumes)
            timestamp = bar_timestamps[-1]  # bar timestamp = last tick in the bar

            bars.append(
                {
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": total_volume,
                }
            )

            # reset accumulators
            cum_dollar = 0
            bar_prices = []
            bar_volumes = []
            bar_timestamps = []

    return pd.DataFrame(bars)
