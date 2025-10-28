import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class NextReturnRegrDataset(Dataset):
    def __init__(self, ohlcv: pd.DataFrame, lookback: int):

        self.lookback = lookback
        close = ohlcv["close"].to_numpy()
        r = np.log(ohlcv["close"]).diff().bfill().to_numpy()

        self.X = (
            np.lib.stride_tricks.sliding_window_view(close, lookback)
            .reshape((-1, lookback, 1))
            .astype(np.float32)
        )

        self.y = np.array(r[lookback:], dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),
            torch.tensor(self.y[i]),
            i,
        )


class NextReturnRegrDataModule(L.LightningDataModule):
    def __init__(
        self,
        data,
        sequence_length: int,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["data"])

        self.data = data
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage=None):

        split_idx = int(len(self.data) * (1 - self.val_split))
        train_data = self.data.iloc[:split_idx]
        val_data = self.data.iloc[split_idx + self.sequence_length :]
        self.train_dataset = NextReturnRegrDataset(
            train_data, lookback=self.sequence_length
        )
        self.val_dataset = NextReturnRegrDataset(
            val_data, lookback=self.sequence_length
        )

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
