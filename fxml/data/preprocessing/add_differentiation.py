from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm


def getWeights(d, lags):
    # return the weights from the series expansion of the differencing operator
    # for real orders d and up to lags coefficients
    w = [1]
    for k in range(1, lags):
        w.append(-w[-1] * ((d - k + 1)) / k)
    w = np.array(w).reshape(-1, 1)
    return w


def cutoff_find(order, cutoff, start_lags):  #
    """
    order: our dearest d
    cutoff: 1e-5 for us
    start_lags: is an initial amount of lags in which the loop will start, this can be set to high values in order to speed up the algo
    """
    val = np.inf
    lags = start_lags
    while abs(val) > cutoff:
        w = getWeights(order, lags)
        val = w[len(w) - 1]
        lags += 1
    return lags


def ts_differencing_tau(series, order, tau):
    # return the time series resulting from (fractional) differencing
    lag_cutoff = cutoff_find(order, tau, 1)  # finding lag cutoff with tau
    weights = getWeights(order, lag_cutoff)
    res = 0
    for k in range(lag_cutoff):
        res += weights[k] * series.shift(k).fillna(0)
    return res[lag_cutoff:]


def main():
    SYMBOL = "USDJPY"
    MINUTES = 1
    START_DATE = "20240101"
    END_DATE = "20241231"
    RESAMPLED_NAME = f"{SYMBOL}-{MINUTES}m-{START_DATE}-{END_DATE}"
    # Base directories
    BASE_DIR = Path("data")
    RESAMPLED_DIR = BASE_DIR / "resampled"
    PROCESSED_DIR = BASE_DIR / "processed"

    RESAMPLED_FILE_PATH = RESAMPLED_DIR / f"{RESAMPLED_NAME}.pkl"
    PROCESSED_FILE_PATH = PROCESSED_DIR / f"{RESAMPLED_NAME}_DIFF.pkl"

    df = pd.read_pickle(RESAMPLED_FILE_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    df_log = pd.DataFrame(np.log(df["close"]), index=df.index)

    df["return"] = df["close"] - df["close"].shift(1)
    df["log_return"] = df_log["close"] - df_log["close"].shift(1)

    adf_score = 99
    adf_threshold = 0.01  # Threshold to define stationarity
    possible_d = np.divide(range(1, 100), 100)
    tau = 1e-4

    for i in tqdm(range(len(possible_d))):
        log_fd = ts_differencing_tau(df_log["close"], possible_d[i], tau)
        adf_score = adfuller(log_fd)[1]
        if adf_score <= adf_threshold:
            df["frac_log_return"] = log_fd
            break

    df.to_pickle(PROCESSED_FILE_PATH)


if __name__ == "__main__":
    main()
