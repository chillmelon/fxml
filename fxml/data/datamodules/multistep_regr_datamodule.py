import lightning as L
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def create_multistep_sequences(
    df,
    lookback: int,
    lookforward: int,
    feat_cols: list[str],
    target_col: list[str],
):
    values = df[feat_cols + target_col].values
    windows = sliding_window_view(values, lookback + lookforward, axis=0).transpose(
        0, 2, 1
    )  # (samples, lookback+lookforward, n_features+1)

    X = windows[:, :lookback, : len(feat_cols)]  # (samples, lookback, n_features)
    y = windows[:, lookback:, len(feat_cols)]  # (samples, lookforward, 1)

    return X, y


class MultiStepRegrDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data,
        test_data,
        feature_cols: list,
        target_col: list,
        lookback: int,
        lookforward: int,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_data", "test_data"])

        self.train_data = train_data
        self.test_data = test_data
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback = lookback
        self.lookforward = lookforward
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage=None):
        # scaler = StandardScaler()
        # self.train_data = pd.DataFrame(
        #     scaler.fit_transform(self.train_data),
        #     index=self.train_data.index,
        #     columns=self.train_data.columns,
        # )
        #
        # self.test_data = pd.DataFrame(
        #     scaler.transform(self.test_data),
        #     index=self.test_data.index,
        #     columns=self.test_data.columns,
        # )

        X_train, y_train = create_multistep_sequences(
            self.train_data,
            self.lookback,
            self.lookforward,
            self.feature_cols,
            self.target_col,
        )

        test_features, test_labels = create_multistep_sequences(
            self.test_data,
            self.lookback,
            self.lookforward,
            self.feature_cols,
            self.target_col,
        )

        X_val, X_test, y_val, y_test = train_test_split(
            test_features, test_labels, test_size=self.val_split, random_state=42
        )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        # Output dataset shapes
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_valid shape: {X_val.shape}, y_valid shape: {y_val.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        self.train_dataset = TensorDataset(X_train, y_train)
        self.valid_dataset = TensorDataset(X_val, y_val)
        self.test_dataset = TensorDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )
