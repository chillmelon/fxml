from os import cpu_count
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import lightning as L
import joblib
from sklearn.preprocessing import StandardScaler

from dataset.forex_clas_dataset import ForexClassificationDataset
from dataset.splitter import Splitter


class ForexClassificationDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 30,
        target: str = "label",
        features: list = ["close_return"],
        target_horizon: int = 1,
        stride: int = 1,
        batch_size: int = 64,
        split_method = None,
        val_split: float = 0.2,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.target = target
        self.features = features
        self.target_horizon = target_horizon
        self.stride = stride

        self.batch_size = batch_size
        self.val_split = val_split
        self.split_method = split_method
        self.num_workers = num_workers
        self.persistent_workers = (True if num_workers > 0 else False)
        self.save_hyperparameters()

    def prepare_data(self):
        self.df = pd.read_pickle(self.data_path)

    def setup(self, stage=None):
        # Now create dataset
        dataset = ForexClassificationDataset(
            data=self.df,
            sequence_length=self.sequence_length,
            horizon=self.target_horizon,
            stride=self.stride,
            features=self.features,
            target=self.target,
            group_col='time_group'
        )

        IDs = dataset.IDs

        splitter = Splitter(
            df=dataset.data,
            IDs=IDs,
            sequence_length=self.sequence_length,
            horizon=self.target_horizon,
            target_col=self.target,
            method=self.split_method,
            test_size=self.val_split,
            random_state=42
        )
        train_indices, val_indices = splitter.split()

        self.train_dataset = Subset(dataset, train_indices)
        self.val_dataset = Subset(dataset, val_indices)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
            shuffle=False
        )
