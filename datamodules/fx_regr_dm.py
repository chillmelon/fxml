import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import lightning as L
import joblib
from sklearn.preprocessing import StandardScaler

from dataset.forex_dataset import ForexDataset

from pathlib import Path


class ForexDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 30,
        target: str = "close",
        features: list = ["close"],
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
        ext = os.path.splitext(self.data_path)[-1]
        if ext == ".csv":
            self.df = pd.read_csv(self.data_path)
        elif ext == ".pkl":
            self.df = pd.read_pickle(self.data_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        # Optionally drop first row if necessary
        self.df = self.df.iloc[1:]  # <-- remove if no longer needed


    def setup(self, stage=None):
        self.df.dropna(subset=self.features, inplace=True)

        if stage == "fit" or stage is None:
            total_size = len(self.df) - self.sequence_length - self.target_horizon + 1
            train_size = int(total_size * (1 - self.val_split))

            # 👇 Training frame (plus buffer for seq len)
            train_df = self.df.iloc[:train_size + self.sequence_length]

            # ✨ Fit scaler on training portion only
            self.scaler = StandardScaler()
            self.scaler.fit(train_df[self.features])

            # ✨ Save scaler for inference or checkpoint
            scaler_path = Path("models/scalers") / f"{Path(self.data_path).stem}_scaler.pkl"
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.scaler, scaler_path)

            # 🚀 Apply to all (not just train_df), important for consistent val/test
            self.df[self.features] = self.scaler.transform(self.df[self.features])

            # Build dataset using normalized df
            dataset = ForexDataset(
                data=self.df,
                sequence_length=self.sequence_length,
                target=self.target,
                target_horizon=self.target_horizon,
                features=self.features
            )

            self.train_dataset = Subset(dataset, list(range(train_size)))
            self.val_dataset = Subset(dataset, list(range(train_size, len(dataset))))


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
