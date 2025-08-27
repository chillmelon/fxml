from collections import Counter

import lightning as L
import torch
from torch.utils.data import DataLoader

from dataset.direction_dataset import DirectionDataset


class ConfidenceDataModule(L.LightningDataModule):
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

    def setup(self, stage=None):
        # sort by time index to maintain temporal order
        sorted_events = self.labels.sort_index()
        n_val = int(len(sorted_events) * self.val_split)

        val_events = sorted_events.iloc[-n_val:]
        train_events = sorted_events.iloc[:-n_val]

        self.train_dataset = DirectionDataset(
            data=self.data,
            events=train_events,
            sequence_length=self.sequence_length,
            features_cols=self.features,
            target_col=self.target,
        )

        self.val_dataset = DirectionDataset(
            data=self.data,
            events=val_events,
            sequence_length=self.sequence_length,
            features_cols=self.features,
            target_col=self.target,
        )

        labels = self.train_dataset.y
        counts = Counter(labels)
        total = sum(counts.values())
        num_classes = len(counts)

        weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
        self.class_weights = torch.tensor(weights, dtype=torch.float32)

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
            pin_memory=(self.num_workers > 0),
        )
