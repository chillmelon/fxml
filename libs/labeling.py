"""
Event-based labeling module for forex trading ML models.

This module contains functions for creating triple-barrier labeling
using event-based sampling for forex price data.
"""

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from ta.volatility import AverageTrueRange
from typing import Optional, List, Tuple


def get_daily_vol(close: pd.Series, span0: int = 100) -> pd.Series:
    """
    Calculate daily volatility using exponential weighted moving average.
    
    Args:
        close: Price series
        span0: EWM span for volatility calculation
        
    Returns:
        Daily volatility series
    """
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(
        close.index[df0 - 1],
        index=close.index[close.shape[0] - df0.shape[0]:]
    )
    
    try:
        df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    except Exception as e:
        print(f"Error in daily vol calculation: {e}")
        print('Adjusting shape of close.loc[df0.index]')
        cut = close.loc[df0.index].shape[0] - close.loc[df0.values].shape[0]
        df0 = close.loc[df0.index].iloc[:-cut] / close.loc[df0.values].values - 1
    
    df0 = df0.ewm(span=span0).std().rename('dailyVol')
    return df0


def get_atr_volatility(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    Calculate ATR-based volatility target.
    
    Args:
        df: OHLC DataFrame
        window: ATR window
        
    Returns:
        ATR return series
    """
    atr = AverageTrueRange(
        high=df['high'], 
        low=df['low'], 
        close=df['close'], 
        window=window
    )
    atr_price = atr.average_true_range().rename(f"atr{window}")
    atr_ret = atr_price / df['close'].shift(1)
    return atr_ret


def get_vertical_barrier(t_events: pd.DatetimeIndex, 
                        close: pd.Series, 
                        delta: pd.Timedelta) -> pd.Series:
    """
    Create vertical barrier (time-based exit) for events.
    
    Args:
        t_events: Event timestamps
        close: Price series
        delta: Time delta for barrier
        
    Returns:
        Series of barrier timestamps
    """
    barrier_times = t_events + delta
    t1_idx = close.index.searchsorted(barrier_times)
    valid_idx = t1_idx[t1_idx < len(close)]
    t1 = pd.Series(
        close.index[valid_idx], 
        index=t_events[:len(valid_idx)]
    )
    return t1


def apply_pt_sl_on_t1(close: pd.Series, events: pd.DataFrame, 
                     ptSl: List[float], molecule: pd.Index) -> pd.DataFrame:
    """
    Apply stop loss/profit taking barriers.
    
    Args:
        close: Price series
        events: Events DataFrame with columns ['t1', 'trgt', 'side']
        ptSl: [profit_taking_multiplier, stop_loss_multiplier]
        molecule: Subset of event indices to process
        
    Returns:
        DataFrame with barrier touch times
    """
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    
    pt = ptSl[0] * events_['trgt'] if ptSl[0] > 0 else pd.Series(index=events.index)
    sl = -ptSl[1] * events_['trgt'] if ptSl[1] > 0 else pd.Series(index=events.index)

    for loc, t1 in events_['t1'].fillna(close.index[-1]).items():
        df0 = close[loc:t1]
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()
        
    return out


def parallel_apply(func, items: pd.Index, num_threads: int = 4, **kwargs) -> pd.DataFrame:
    """
    Apply function in parallel across chunks of items.
    
    Args:
        func: Function to apply
        items: Items to process
        num_threads: Number of threads
        **kwargs: Additional arguments for func
        
    Returns:
        Concatenated results
    """
    def worker(molecule):
        return func(molecule=molecule, **kwargs)

    chunks = np.array_split(items, num_threads)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(worker, chunks))

    return pd.concat(results).sort_index()


def get_events(close: pd.Series, 
              t_events: pd.DatetimeIndex,
              ptSl: List[float],
              trgt: pd.Series,
              min_ret: float,
              num_threads: int = 4,
              t1: Optional[pd.Series] = None,
              side: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Create triple-barrier events.
    
    Args:
        close: Price series
        t_events: Event timestamps
        ptSl: [profit_taking_multiplier, stop_loss_multiplier]
        trgt: Volatility target series
        min_ret: Minimum return threshold
        num_threads: Number of threads for parallel processing
        t1: Vertical barrier series (optional)
        side: Position side series (optional)
        
    Returns:
        Events DataFrame with barrier information
    """
    # Filter targets
    trgt = trgt.loc[t_events]
    trgt = trgt[trgt > min_ret]

    # Set vertical barrier
    if t1 is None:
        t1 = pd.Series(pd.NaT, index=t_events)

    # Build events DataFrame
    if side is None:
        side_, ptSl_ = pd.Series(1., index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        side_, ptSl_ = side.loc[trgt.index], ptSl[:2]

    events = pd.concat(
        {'t1': t1, 'trgt': trgt, 'side': side_}, 
        axis=1
    ).dropna(subset=['trgt'])

    # Apply barriers in parallel
    df0 = parallel_apply(
        func=apply_pt_sl_on_t1,
        items=events.index,
        num_threads=num_threads,
        close=close,
        events=events,
        ptSl=ptSl_
    )

    # Choose the first touched barrier
    events['t1'] = df0.dropna(how='all').min(axis=1)
    
    if side is None:
        events = events.drop('side', axis=1)
        
    return events


def get_bins(events: pd.DataFrame, 
            close: pd.Series, 
            t1: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Compute event outcomes and create classification labels.
    
    Args:
        events: Events DataFrame with 't1', 'trgt', and optional 'side'
        close: Price series
        t1: Original vertical barrier series (for proper labeling)
        
    Returns:
        DataFrame with returns and binary labels
    """
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    
    if 'side' in events_: 
        out['ret'] *= events_['side']  # meta-labeling
    
    out['bin'] = np.sign(out['ret'])

    if 'side' not in events_ and t1 is not None:
        # Set to 0 when vertical barrier is touched
        vtouch_first_idx = events[events['t1'].isin(t1.values)].index
        out.loc[vtouch_first_idx, 'bin'] = 0.

    if 'side' in events_: 
        out.loc[out['ret'] <= 0, 'bin'] = 0  # meta-labeling
        
    return out


def get_concurrency(events: pd.DataFrame, price_index: pd.DatetimeIndex) -> pd.Series:
    """
    Calculate concurrency: number of overlapping events at each time.
    
    Args:
        events: Events DataFrame with 't1' column
        price_index: Full time index from OHLCV data
        
    Returns:
        Concurrency count series indexed by time
    """
    concurrency = pd.Series(0, index=price_index)

    for start, end in events['t1'].items():
        concurrency[start:end] += 1

    return concurrency


def create_direction_labels(df: pd.DataFrame,
                          t_events: pd.DatetimeIndex,
                          atr_window: int = 20,
                          barrier_delta_minutes: int = 120,
                          pt_sl_multipliers: Tuple[float, float] = (1.0, 1.0),
                          min_ret_factor: float = 0.5,
                          num_threads: int = 4,
                          intraday_only: bool = True) -> pd.DataFrame:
    """
    Complete pipeline to create direction labels from OHLC data and events.
    
    Args:
        df: OHLC DataFrame with columns ['open', 'high', 'low', 'close']
        t_events: Event timestamps
        atr_window: Window for ATR volatility calculation
        barrier_delta_minutes: Minutes for vertical barrier
        pt_sl_multipliers: (profit_taking, stop_loss) multipliers
        min_ret_factor: Minimum return threshold as factor of mean target
        num_threads: Number of threads for parallel processing
        intraday_only: Keep only intraday events
        
    Returns:
        Labeled events DataFrame
    """
    # Calculate volatility target
    atr_ret = get_atr_volatility(df, window=atr_window)
    trgt = atr_ret.shift(1).reindex(t_events, method='ffill')
    
    # Create vertical barrier
    t1 = get_vertical_barrier(
        t_events, 
        df['close'], 
        delta=pd.Timedelta(minutes=barrier_delta_minutes)
    )
    
    # Set minimum return threshold
    min_ret = trgt.mean() * min_ret_factor
    
    # Get events with triple barriers
    events = get_events(
        close=df['close'],
        t_events=t_events,
        ptSl=list(pt_sl_multipliers),
        trgt=trgt,
        min_ret=min_ret,
        num_threads=num_threads,
        t1=t1,
        side=None
    )
    
    # Filter to valid events
    events = events.dropna().copy()
    
    # Keep only intraday events if requested
    if intraday_only:
        events = events[events.index.date == events.t1.dt.date]
    
    # Create classification labels
    labels = get_bins(events, df['close'], t1=t1)
    
    # Add class labels (0, 1, 2 instead of -1, 0, 1)
    labels["bin_class"] = labels["bin"] + 1
    
    # Join events and labels
    labeled_events = events.join(labels, how='inner')
    
    return labeled_events