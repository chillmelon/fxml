import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import lightning as L
import joblib
from sklearn.preprocessing import StandardScaler

from dataset.forex_dataset import ForexDataset


class ForexDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 30,
        target: str = "close_pct_delta",
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
        self.df = pd.read_pickle(self.data_path)

    def setup(self, stage=None):
        # ⚡ Only fit scaler on training data
        if stage == "fit" or stage is None:
            dataset = ForexDataset(
                data=self.df,
                sequence_length=self.sequence_length,
                target=self.target,
                target_horizon=self.target_horizon,
                features=self.features
            )
            total_size = len(dataset)
            train_size = int(total_size * (1 - self.val_split))
            val_size = total_size - train_size

            # Chronological split
            self.train_dataset = Subset(dataset, list(range(train_size)))
            self.val_dataset = Subset(dataset, list(range(train_size, total_size)))

            # ✨ Normalize only based on train data
            train_df = self.df.iloc[:train_size]

            self.scaler = StandardScaler()
            self.scaler.fit(train_df[self.features])

            # Apply normalization
            self.df[self.features] = self.scaler.transform(self.df[self.features])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
