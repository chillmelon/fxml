import lightning as L
from torch.utils.data import DataLoader, Subset
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from dataset.event_based_dataset import EventBasedDataset

class EventBasedDataModule(L.LightningDataModule):
    def __init__(
        self,
        data,
        labels,
        sequence_length: int,
        features: list,
        target: str,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split: float = 0.2,
        shuffle: bool = True,
        random_state: int = 42
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["data", "labels"])
        self.data = data[features]
        self.labels = labels[target]
        self.sequence_length = sequence_length
        self.features = features
        self.target = target
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.shuffle = shuffle
        self.random_state = random_state

    def setup(self, stage=None):
        # sort by time index to maintain temporal order
        sorted_events = self.labels.sort_index()
        n_val = int(len(sorted_events) * self.val_split)

        val_events = sorted_events.iloc[-n_val:]
        train_events = sorted_events.iloc[:-n_val]

        self.train_dataset = EventBasedDataset(
            data=self.data,
            events=train_events,
            sequence_length=self.sequence_length,
            features_cols=self.features,
            target_col=self.target,
        )

        self.val_dataset = EventBasedDataset(
            data=self.data,
            events=val_events,
            sequence_length=self.sequence_length,
            features_cols=self.features,
            target_col=self.target,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.num_workers > 0),
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.num_workers > 0),
            persistent_workers=(self.num_workers > 0),
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.num_workers > 0)
        )
