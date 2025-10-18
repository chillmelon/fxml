import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta


def ema_cross(data, fast=12, slow=26):

    ema_fast = ta.ema(data["close"], length=fast)
    ema_slow = ta.ema(data["close"], length=slow)

    signal = pd.Series(np.full(len(data), np.nan), index=data.index)

    signal.loc[ema_fast > ema_slow] = 1
    signal.loc[ema_fast < ema_slow] = -1
    return signal


if __name__ == "__main__":

    # Load data
    data = pd.read_pickle("data/resampled/EURUSD-60m-20240101-20241231.pkl")
    data["timestamp"] = data["timestamp"].astype("datetime64[s]")
    data = data.set_index("timestamp")

    # Trendline parameter
    fast = 5
    slow = 50

    data["signal"] = ema_cross(data, fast, slow)
    data["r"] = np.log(data["close"]).diff().shift(-1)
    strat_r = data["signal"] * data["r"]

    pf = strat_r[strat_r > 0].sum() / strat_r[strat_r < 0].abs().sum()
    print("Profit Factor", fast, slow, pf)

    plt.style.use("dark_background")
    strat_r.cumsum().plot()
    plt.ylabel("Cumulative Log Return")
    plt.show()
