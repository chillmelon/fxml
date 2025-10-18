import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class NextBarRegrDataset(Dataset):
    def __init__(self, ohlcv: pd.DataFrame, lookback: int):

        self.lookback = lookback
        close = np.log(ohlcv["close"].to_numpy())
        len_df = len(close)
        list_slices = [close[i : len_df - lookback + i] for i in range(0, lookback)]
        self.X = np.array(
            np.vstack(list_slices).T.reshape((-1, lookback, 1)), dtype=np.float32
        )
        self.y = np.array(close[lookback:], dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),
            torch.tensor(self.y[i]),
            i,
        )
