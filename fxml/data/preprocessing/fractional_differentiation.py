import numpy as np
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


def find_optimal_fraction(series, tau, adf_threshold):
    adf_score = 99
    possible_d = np.divide(range(1, 100), 100)

    for i in tqdm(range(len(possible_d))):
        log_fd = ts_differencing_tau(series, possible_d[i], tau)
        adf_score = adfuller(log_fd)[1]
        if adf_score < adf_threshold:
            print(f"Found optimal d = {possible_d[i]:.2f}")
            print(f"ADF p-value = {adf_score:.6f}")
            return possible_d[i], log_fd
    return None, None
