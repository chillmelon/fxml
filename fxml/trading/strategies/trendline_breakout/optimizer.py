import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fxml.trading.strategies.trendline_breakout.trendline_breakout import (
    trendline_breakout,
)


def main():
    data = pd.read_pickle("data/resampled/EURUSD-60m-20240101-20241231.pkl")
    data["timestamp"] = data["timestamp"].astype("datetime64[s]")
    data = data.set_index("timestamp")
    data = data.dropna()

    ## Parameter Sweep
    lookbacks = list(range(24, 169, 2))
    pfs = []

    lookback_returns = pd.DataFrame()
    for lookback in lookbacks:
        support, resist, signal = trendline_breakout(data["close"].to_numpy(), lookback)
        data["signal"] = signal

        data["r"] = np.log(data["close"]).diff().shift(-1)
        strat_r = data["signal"] * data["r"]

        pf = strat_r[strat_r > 0].sum() / strat_r[strat_r < 0].abs().sum()
        print("Profit Factor", lookback, pf)
        pfs.append(pf)

        lookback_returns[lookback] = strat_r

    plt.style.use("dark_background")
    x = pd.Series(pfs, index=lookbacks)
    x.plot()
    plt.ylabel("Profit Factor")
    plt.xlabel("Trendline Lookback")
    plt.axhline(1.0, color="white")
    plt.show()


if __name__ == "__main__":
    main()
