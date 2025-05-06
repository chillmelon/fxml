import lightning as L
from torch.utils.data import DataLoader, Subset
import pandas as pd

from dataset.dataset import ForexDataset

class ForexDataModule(L.LightningDataModule):
    def __init__(
        self,
        data,
        IDs: list,
        sequence_length: int,
        horizon: int,
        features: list,
        target: list,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split: float = 0.2,
        shuffle: bool = True,
        random_state: int = 42
    ):
        super().__init__()
        self.data = data
        self.IDs = IDs
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.features = features
        self.target = target
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.shuffle = shuffle
        self.random_state = random_state

    def setup(self, stage=None):
        from sklearn.model_selection import train_test_split

        train_idx, val_idx = train_test_split(
            self.IDs,
            test_size=self.val_split,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

        self.train_dataset = ForexDataset(
            self.data, train_idx, self.sequence_length, self.horizon, self.features, self.target
        )

        self.val_dataset = ForexDataset(
            self.data, val_idx, self.sequence_length, self.horizon, self.features, self.target
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
