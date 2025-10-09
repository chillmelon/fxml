import numpy as np
import pandas as pd
from tqdm import tqdm


def cusum_filter(closes: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """
    Extract events using CUSUM (Cumulative Sum) filter.

    The CUSUM filter detects events when cumulative positive or negative
    movements exceed a specified threshold. This is useful for identifying
    significant directional movements in financial time series.

    Args:
        closes : Close prices
        threshold: Threshold for CUSUM filter (positive value)

    Returns:
        DatetimeIndex of event timestamps where events were detected
    """
    if threshold <= 0:
        raise ValueError("Threshold must be positive")

    if len(closes) == 0:
        return pd.DatetimeIndex([])

    values = closes.values
    timestamps = closes.index

    # Pre-allocate arrays for performance
    event_mask = np.zeros(len(values), dtype=bool)

    # Initialize cumulative sums
    cum_pos, cum_neg = 0.0, 0.0

    # Process each value
    for i in tqdm(
        range(len(values)), desc="Processing CUSUM events", disable=len(values) < 1000
    ):
        # Update cumulative sums
        cum_pos = max(0.0, cum_pos + values[i])
        cum_neg = min(0.0, cum_neg + values[i])

        # Check for positive threshold breach
        if cum_pos > threshold:
            event_mask[i] = True
            cum_pos = 0.0

        # Check for negative threshold breach
        if cum_neg < -threshold:
            event_mask[i] = True
            cum_neg = 0.0

    return timestamps[event_mask]


def z_score_filter(
    closes: pd.Series,
    mean_window: int,
    std_window: int,
    z_score: float = 3,
) -> pd.DatetimeIndex:
    """
    Filter which implements z_score filter.

    Parameters
    ----------
    closes : pd.Series
        Close prices
    mean_window : int
        Rolling mean window
    std_window : int
        Rolling std window
    z_score : float
        Number of standard deviations to trigger the event

    Returns
    -------
    t_events : pd.DatetimeIndex or list
        Vector of datetimes when the events occurred. This is used later to sample.

    Notes
    -----
    Reference: Implement the idea of z-score filter here at
    [StackOverflow Question]
    (https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data).
    """
    t_events = closes[
        closes
        >= closes.rolling(window=mean_window).mean()
        + z_score * closes.rolling(window=std_window).std()
    ].index
    event_timestamps = pd.DatetimeIndex(t_events)
    return event_timestamps
