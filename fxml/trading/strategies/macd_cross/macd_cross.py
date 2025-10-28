from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from fxml.trading.strategies.macd_cross.macd_indicator import macd


def macd_cross_strategy(
    close: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
):
    # Calculate MACD components
    macd_line, signal_line, histogram = macd(
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


def optimize_macd(ohlcv):
    fast_periods = list(range(5, 21, 2))
    slow_periods = list(range(20, 100, 2))
    signal_periods = list(range(6, 15, 2))

    results = []
    best_pf = 0
    best_params = (0, 0, 0)

    for fast, slow, signal in tqdm(
        product(fast_periods, slow_periods, signal_periods),
        total=len(fast_periods) * len(slow_periods) * len(signal_periods),
    ):
        if fast >= slow:
            continue
        _, _, _, sig = macd_cross_strategy(
            ohlcv["close"].to_numpy(), fast, slow, signal
        )
        r = np.log(ohlcv["close"]).diff().shift(-1)
        strat_r = sig * r

        pf = strat_r[strat_r > 0].sum() / strat_r[strat_r < 0].abs().sum()
        if pf > best_pf:
            best_pf = pf
            best_params = (fast, slow, signal)
        results.append(
            {
                "fast": fast,
                "slow": slow,
                "signal": signal,
                "profit_factor": pf,
            }
        )

    return best_params, best_pf, results


def main():
    data = pd.read_pickle("data/resampled/EURUSD-15m-20240101-20241231.pkl")
    data["timestamp"] = data["timestamp"].astype("datetime64[s]")
    data = data.set_index("timestamp")
    data = data.dropna()

    (fast, slow, signal), best_pf, results = optimize_macd(data)

    macd_line, signal_line, histogram, sig = macd_cross_strategy(
        data["close"].to_numpy(), fast, slow, signal
    )
    data["macd"] = macd_line
    data["signal_line"] = signal_line
    data["histogram"] = histogram
    data["signal"] = sig
    data["r"] = np.log(data["close"]).diff().shift(-1)
    strat_r = data["signal"] * data["r"]

    pf = strat_r[strat_r > 0].sum() / strat_r[strat_r < 0].abs().sum()
    print("Profit Factor", pf)
    print("Best Parameters", (fast, slow, signal))

    plt.style.use("dark_background")
    strat_r.cumsum().plot()
    plt.ylabel("Cumulative Log Return")
    plt.show()


if __name__ == "__main__":
    main()
