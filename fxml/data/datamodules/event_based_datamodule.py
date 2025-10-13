import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from fxml.data.datasets.direction_dataset import DirectionDataset


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
        random_state: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["data", "labels"])
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length
        self.features = features
        self.target = target
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.shuffle = shuffle
        self.random_state = random_state

        self.class_weights = None
        self._train_sample_weights = None

    def setup(self, stage=None):
        print("====== Start Setting Up Data Module =====")
        # sort by time index to maintain temporal order
        sorted_events = self.labels.sort_index()
        n_val = int(len(sorted_events) * self.val_split)

        val_events = sorted_events.iloc[-n_val:]
        train_events = sorted_events.iloc[:-n_val]

        print("====== Start Building Training Dataset =====")
        self.train_dataset = DirectionDataset(
            data=self.data,
            labels=train_events,
            sequence_length=self.sequence_length,
            features_cols=self.features,
            target_col=self.target,
        )
        print(self.train_dataset[1])

        print("====== Start Building Validation Dataset =====")
        self.val_dataset = DirectionDataset(
            data=self.data,
            labels=val_events,
            sequence_length=self.sequence_length,
            features_cols=self.features,
            target_col=self.target,
        )
        print(self.val_dataset[1])

        print("====== End Setting Up Data Module =====")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
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
            pin_memory=(self.num_workers > 0),
        )
