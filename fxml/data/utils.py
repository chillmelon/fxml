import numpy as np
import pandas as pd


def create_sequences(
    df: pd.DataFrame,
    lookback: int,
    stride: int,
    feat_cols: list[str],
    target_col: str,
):
    len_df = df.shape[0]
    feature_values = df[feat_cols].values
    target_values = df[target_col].values
    X = []
    y = []
    for i in range(0, len_df - lookback, stride):
        idx_start, idx_end = i, i + lookback
        X.append(feature_values[idx_start:idx_end])
    for i in range(lookback, len_df, stride):
        y.append(target_values[i])

    return np.array(X), np.array(y)
