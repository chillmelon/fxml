import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class NextBarRegrDataset(Dataset):
    def __init__(self, close: np.ndarray, lookback: int):

        self.lookback = lookback
        # close = np.log(ohlcv["close"].to_numpy())
        len_df = len(close)
        self.X = (
            np.lib.stride_tricks.sliding_window_view(close, lookback)
            .reshape((-1, lookback, 1))
            .astype(np.float32)
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


class NextBarDataRegrModule(L.LightningDataModule):
    def __init__(
        self,
        data,
        features: list,
        target: str,
        sequence_length: int,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["data"])

        self.features = features
        self.target = target

        self.data = data
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage=None):

        split_idx = int(len(self.data) * (1 - self.val_split))
        train_data = self.data[self.features + [self.target]].iloc[:split_idx]
        val_data = self.data[self.features + [self.target]].iloc[
            split_idx + self.sequence_length :
        ]
        self.train_dataset = NextBarRegrDataset(
            train_data, lookback=self.sequence_length
        )
        self.val_dataset = NextBarRegrDataset(val_data, lookback=self.sequence_length)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
