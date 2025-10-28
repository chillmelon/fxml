import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import pandas_ta as ta
from tqdm import tqdm

from fxml.trading.strategies.trendline_breakout.trendline_automation import (
    fit_trendlines_single,
)


def trendline_breakout(close: np.ndarray, lookback: int):
    s_tl = np.zeros(len(close))
    s_tl[:] = np.nan

    r_tl = np.zeros(len(close))
    r_tl[:] = np.nan

    sig = np.zeros(len(close))

    for i in range(lookback, len(close)):
        # NOTE window does NOT include the current candle
        window = close[i - lookback : i]

        s_coefs, r_coefs = fit_trendlines_single(window)

        # Find current value of line, projected forward to current bar
        s_val = s_coefs[1] + lookback * s_coefs[0]
        r_val = r_coefs[1] + lookback * r_coefs[0]

        s_tl[i] = s_val
        r_tl[i] = r_val

        if close[i] > r_val:
            sig[i] = 1.0
        elif close[i] < s_val:
            sig[i] = -1.0
        else:
            sig[i] = sig[i - 1]

    return s_tl, r_tl, sig


def optimize_trendline(data):
    lookbacks = list(range(4, 180, 2))

    best_pf = 0.0
    best_lookback = -1
    r = np.log(data["close"]).diff().shift(-1)

    for lookback in tqdm(lookbacks, position=1):
        _, _, signal = trendline_breakout(data["close"].to_numpy(), lookback)

        sig_rets = signal * r

        sig_pf = sig_rets[sig_rets > 0].sum() / sig_rets[sig_rets < 0].abs().sum()
        if sig_pf > best_pf:
            best_pf = sig_pf
            best_lookback = lookback

    return best_lookback, best_pf


def main():
    data = pd.read_pickle("data/resampled/EURUSD-60m-20240101-20241231.pkl")
    data["timestamp"] = data["timestamp"].astype("datetime64[s]")
    data = data.set_index("timestamp")
    data = data.dropna()

    lookback, _ = optimize_trendline(data)
    support, resist, signal = trendline_breakout(data["close"].to_numpy(), lookback)
    data["support"] = support
    data["resist"] = resist
    data["signal"] = signal

    plt.style.use("dark_background")
    data["close"].plot(label="Close")
    data["resist"].plot(label="Resistance", color="green")
    data["support"].plot(label="Support", color="red")
    plt.show()

    data["r"] = np.log(data["close"]).diff().shift(-1)
    strat_r = data["signal"] * data["r"]

    pf = strat_r[strat_r > 0].sum() / strat_r[strat_r < 0].abs().sum()
    print("Profit Factor", lookback, pf)

    strat_r.cumsum().plot()
    plt.ylabel("Cumulative Log Return")
    plt.show()


if __name__ == "__main__":
    main()
