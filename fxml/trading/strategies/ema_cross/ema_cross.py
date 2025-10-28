from itertools import product

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


def optimize_ema_cross(data):
    fast_range = range(2, 55, 2)  # 8, 10, 12, 14, 16, 18
    slow_range = range(24, 169, 2)  # 5, 7, 9, 11, 13

    r = np.log(data["close"]).diff().shift(-1)

    best_pf = 0
    best_fast = -1
    best_slow = -1

    for fast, slow in product(fast_range, slow_range):
        # Skip invalid combinations where fast >= slow
        if fast >= slow:
            continue
        signal = ema_cross(data, fast, slow)
        sig_rets = signal * r
        sig_pf = sig_rets[sig_rets > 0].sum() / sig_rets[sig_rets < 0].abs().sum()

        if sig_pf > best_pf:
            best_pf = sig_pf
            best_fast = fast
            best_slow = slow

    return (best_fast, best_slow), best_pf


def walkforward_ema_cross(
    ohlc: pd.DataFrame, train_lookback: int = 24 * 365 * 4, train_step: int = 24 * 30
):

    n = len(ohlc)
    wf_signal = np.full(n, np.nan)
    tmp_signal = pd.Series(np.full(n, np.nan), index=ohlc.index)

    next_train = train_lookback
    for i in range(next_train, n):
        if i == next_train:
            best_params, _ = optimize_ema_cross(ohlc.iloc[i - train_lookback : i])
            tmp_signal = ema_cross(
                ohlc.iloc[i - train_lookback : i], best_params[0], best_params[1]
            )
            next_train += train_step

        wf_signal[i] = tmp_signal.iloc[i]

    return wf_signal


if __name__ == "__main__":

    # Load data
    data = pd.read_pickle("data/resampled/EURUSD-60m-20240101-20241231.pkl")
    data["timestamp"] = data["timestamp"].astype("datetime64[s]")
    data = data.set_index("timestamp")

    # Trendline parameter
    (fast, slow), _ = optimize_ema_cross(data)

    data["signal"] = ema_cross(data, fast, slow)
    data["r"] = np.log(data["close"]).diff().shift(-1)
    strat_r = data["signal"] * data["r"]

    pf = strat_r[strat_r > 0].sum() / strat_r[strat_r < 0].abs().sum()
    print("Profit Factor", fast, slow, pf)

    plt.style.use("dark_background")
    strat_r.cumsum().plot()
    plt.ylabel("Cumulative Log Return")
    plt.show()
