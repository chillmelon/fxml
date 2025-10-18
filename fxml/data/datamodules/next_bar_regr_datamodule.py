import lightning as L
from torch.utils.data import DataLoader

from fxml.data.datasets.next_bar_regr_dataset import NextBarRegrDataset


class NextBarDataRegrModule(L.LightningDataModule):
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

        train_data = self.data.iloc[: int(len(self.data) * (1 - self.val_split))]
        val_data = self.data.iloc[int(len(self.data) * (1 - self.val_split)) :]
        self.train_dataset = NextBarRegrDataset(
            train_data, lookback=self.sequence_length
        )
        print(self.train_dataset[:2])
        self.val_dataset = NextBarRegrDataset(val_data, lookback=self.sequence_length)

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
