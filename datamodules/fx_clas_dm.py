from os import cpu_count
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import lightning as L
import joblib
from sklearn.preprocessing import StandardScaler

from dataset.forex_clas_dataset import ForexClassificationDataset


class ForexClassificationDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 30,
        target: str = "label",
        features: list = ["close_pct_delta"],
        target_horizon: int = 1,
        batch_size: int = 64,
        val_split: float = 0.2
    ):
        super().__init__()
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.target = target
        self.features = features
        self.target_horizon = target_horizon
        self.batch_size = batch_size
        self.val_split = val_split
        self.save_hyperparameters()

    def prepare_data(self):
        self.df = pd.read_csv(self.data_path)[1:]

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # ✨ Normalize only based on train data
            total_size = len(self.df) - self.sequence_length - self.target_horizon + 1
            train_size = int(total_size * (1 - self.val_split))

            train_df = self.df.iloc[:train_size + self.sequence_length]

            self.scaler = StandardScaler()
            self.scaler.fit(train_df[self.features])

            # Apply normalization to the entire feature column
            self.df[self.features] = self.scaler.transform(self.df[self.features])

            # Now create dataset
            dataset = ForexClassificationDataset(
                data=self.df,
                sequence_length=self.sequence_length,
                target=self.target,
                features=self.features
            )

            self.train_dataset = Subset(dataset, list(range(train_size)))
            self.val_dataset = Subset(dataset, list(range(train_size, len(dataset))))


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            # num_workers=cpu_count(),
            # persistent_workers=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            # num_workers=cpu_count(),
            # persistent_workers=True,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            # num_workers=cpu_count(),
            # persistent_workers=True,
            shuffle=False
        )
