import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def donchian_breakout(ohlc: pd.DataFrame, lookback: int):
    # input df is assumed to have a 'close' column
    upper = ohlc["close"].rolling(lookback - 1).max().shift(1)
    lower = ohlc["close"].rolling(lookback - 1).min().shift(1)
    signal = pd.Series(np.full(len(ohlc), np.nan), index=ohlc.index)
    signal.loc[ohlc["close"] > upper] = 1
    signal.loc[ohlc["close"] < lower] = -1
    signal = signal.ffill()
    return signal


if __name__ == "__main__":

    # Load data
    data = pd.read_pickle("data/resampled/EURUSD-60m-20240101-20241231.pkl")
    data["timestamp"] = data["timestamp"].astype("datetime64[s]")
    data = data.set_index("timestamp")

    # Trendline parameter
    lookback = 24

    data["signal"] = donchian_breakout(data, lookback)
    data["r"] = np.log(data["close"]).diff().shift(-1)
    strat_r = data["signal"] * data["r"]

    pf = strat_r[strat_r > 0].sum() / strat_r[strat_r < 0].abs().sum()
    print("Profit Factor", lookback, pf)

    plt.style.use("dark_background")
    strat_r.cumsum().plot()
    plt.ylabel("Cumulative Log Return")
    plt.show()
