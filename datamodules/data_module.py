import lightning as L
from torch.utils.data import DataLoader, Subset
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from dataset.dataset import ForexDataset

class ForexDataModule(L.LightningDataModule):
    def __init__(
        self,
        data,
        sequence_length: int,
        features: list,
        target: list,
        stride: int = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split: float = 0.2,
        shuffle: bool = True,
        random_state: int = 42
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["data"])
        self.data = data[features + target]
        self.sequence_length = sequence_length
        self.features = features
        self.target = target
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.shuffle = shuffle
        self.random_state = random_state

    def setup(self, stage=None):
        # data_clean= self.data.copy()
        # Stratified split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.val_split, random_state=self.random_state)
        y = self.data[self.target]
        train_idx, val_idx = next(sss.split(self.data, y))

        train_df = self.data.iloc[train_idx].reset_index(drop=True)
        val_df = self.data.iloc[val_idx].reset_index(drop=True)

        self.train_dataset = ForexDataset(train_df, self.sequence_length, self.features, self.target, self.stride)
        self.val_dataset = ForexDataset(val_df, self.sequence_length, self.features, self.target)


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
