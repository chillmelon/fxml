#!/usr/bin/env python
# coding: utf-8

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor

# ===========================
# Configuration Parameters
# ===========================
SOURCE = "dukascopy"
SYMBOL = "usdjpy"
MINUTES = 1
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
LOOKBACK = 30
CUSUM_THRESHOLD = 5e-3
PT_SL = [5e-3, 5e-3]
VERTICAL_BARRIER_DAYS = 1
MIN_RET = 0
SPAN_VOL = 100

# ===========================
# File Paths
# ===========================
BASE_NAME = f"{SOURCE}-{SYMBOL}-tick-{START_DATE}-{END_DATE}"
RESAMPLED_NAME = f"{SOURCE}-{SYMBOL}-{MINUTES}m-{START_DATE}-{END_DATE}"
BASE_DIR = Path("../data")
RAW_DIR = BASE_DIR / "raw"
RESAMPLED_DIR = BASE_DIR / "resampled"
LABEL_DIR = BASE_DIR / "labeled"
LABEL_DIR.mkdir(parents=True, exist_ok=True)

RESAMPLED_FILE_PATH = RESAMPLED_DIR / f"{RESAMPLED_NAME}.pkl"

# ===========================
# Utility Functions
# ===========================
def load_data(path):
    df = pd.read_pickle(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['log_volume'] = np.log1p(df['volume'])
    return df.dropna()

def get_tevents(data, threshold):
    values = data.values
    timestamps = data.index
    cum_pos, cum_neg = 0.0, 0.0
    t_events_mask = np.zeros_like(values, dtype=bool)

    for i in tqdm(range(len(values))):
        cum_pos = max(0.0, cum_pos + values[i])
        cum_neg = min(0.0, cum_neg + values[i])
        if cum_pos > threshold:
            t_events_mask[i] = True
            cum_pos = 0.0
        if cum_neg < -threshold:
            t_events_mask[i] = True
            cum_neg = 0.0

    return timestamps[t_events_mask]

def get_side_from_trendline(close, t_events, lookback):
    side = pd.Series(index=t_events, dtype='float32')
    for t in t_events:
        if t not in close.index:
            continue
        try:
            end_loc = close.index.get_loc(t)
            start_loc = end_loc - lookback + 1
            if start_loc < 0:
                continue
            window = close.iloc[start_loc:end_loc + 1]
            x = np.arange(len(window)).reshape(-1, 1)
            y = window.values.reshape(-1, 1)
            model = LinearRegression().fit(x, y)
            trend_value = model.predict([[len(window) - 1]])[0][0]
            side[t] = 1 if y[-1][0] > trend_value else -1
        except Exception:
            continue
    return side.dropna()

def get_daily_vol(close, span):
    idx = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    idx = idx[idx > 0]
    df0 = pd.Series(close.index[idx - 1], index=close.index[-len(idx):])
    try:
        returns = close.loc[df0.index] / close.loc[df0.values].values - 1
    except Exception:
        cut = close.loc[df0.index].shape[0] - close.loc[df0.values].shape[0]
        returns = close.loc[df0.index].iloc[:-cut] / close.loc[df0.values].values - 1
    return returns.ewm(span=span).std().rename('dailyVol')

def apply_pt_sl_on_t1(close, events, pt_sl, molecule):
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    pt = pt_sl[0] * events_['trgt'] if pt_sl[0] > 0 else pd.Series(index=events.index)
    sl = -pt_sl[1] * events_['trgt'] if pt_sl[1] > 0 else pd.Series(index=events.index)
    for loc, t1_ in events_['t1'].fillna(close.index[-1]).items():
        df0 = (close[loc:t1_] / close[loc] - 1) * events_.at[loc, 'side']
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()
    return out

def parallel_apply(func, items, num_threads=4, **kwargs):
    chunks = np.array_split(items, num_threads)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(lambda chunk: func(molecule=chunk, **kwargs), chunks))
    return pd.concat(results).sort_index()

def get_events(close, t_events, pt_sl, trgt, min_ret, num_threads=4, t1=None, side=None):
    trgt = trgt.loc[t_events]
    trgt = trgt[trgt > min_ret]
    if t1 is None:
        t1 = pd.Series(pd.NaT, index=t_events)
    side_, pt_sl_ = (side.loc[trgt.index], pt_sl[:2]) if side is not None else (pd.Series(1., index=trgt.index), [pt_sl[0], pt_sl[0]])
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
    df0 = parallel_apply(apply_pt_sl_on_t1, events.index, num_threads=num_threads, close=close, events=events, pt_sl=pt_sl_)
    events['t1'] = df0.dropna(how='all').min(axis=1)
    if side is None:
        events = events.drop('side', axis=1)
    return events

def get_bins(events, close, t1=None):
    events_ = events.dropna(subset=['t1'])
    px = close.reindex(events_.index.union(events_['t1'].values).drop_duplicates(), method='bfill')
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    if 'side' in events_:
        out['ret'] *= events_['side']
    out['bin'] = np.sign(out['ret'])
    if 'side' not in events_ and t1 is not None:
        vtouch_first_idx = events[events['t1'].isin(t1.values)].index
        out.loc[vtouch_first_idx, 'bin'] = 0.0
    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0.0
    return out

def save_results(events_df, config_dict, label_dir, event_name):
    events_path = label_dir / f"{event_name}_events.parquet"
    config_path = label_dir / f"{event_name}_config.yml"
    events_df.to_parquet(events_path)
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)
    print(f"✅ Labeled events saved to: {events_path}")
    print(f"🧾 Config saved to: {config_path}")

# ===========================
# Pipeline Execution
# ===========================
df = load_data(RESAMPLED_FILE_PATH)
close = df['close']
t_events = get_tevents(df['log_return'], CUSUM_THRESHOLD)
sides = get_side_from_trendline(close, t_events, LOOKBACK)
daily_vol = get_daily_vol(close, SPAN_VOL)
trgt = daily_vol.reindex(t_events, method='ffill')
t1_index = close.index.searchsorted(t_events + pd.Timedelta(days=VERTICAL_BARRIER_DAYS))
t1_index = t1_index[t1_index < len(close)]
t1 = pd.Series(close.index[t1_index], index=t_events[:len(t1_index)])
events = get_events(close, t_events, PT_SL, trgt, MIN_RET, t1=t1, side=sides)
labels = get_bins(events, close, t1=t1)
labeled_events = events.join(labels, how='inner')

# ===========================
# Save Results
# ===========================
event_name = f"cusum_trendlookback{LOOKBACK}_{MINUTES}m_barrier005"
config = {
    'event_name': event_name,
    'source': SOURCE,
    'symbol': SYMBOL,
    'timeframe_minutes': MINUTES,
    'start_date': START_DATE,
    'end_date': END_DATE,
    'event_type': 'CUSUM',
    'cusum_threshold': CUSUM_THRESHOLD,
    'trend_lookback': LOOKBACK,
    'barrier': {
        'take_profit': PT_SL[0],
        'stop_loss': PT_SL[1],
        'vertical_barrier_days': VERTICAL_BARRIER_DAYS,
        'min_target_return': MIN_RET
    },
    'volatility_lookback_span': SPAN_VOL
}
save_results(labeled_events, config, LABEL_DIR, event_name)
