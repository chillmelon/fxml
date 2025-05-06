from pandas.core.generic import dt
import torch
from torch.utils.data import Dataset
import numpy as np

class ForexDataset(Dataset):
    """Dataset for sequence classification/forecasting with multi-step horizon."""

    def __init__(self, data, IDs, sequence_length, horizon, features, target):
        self.data = data
        self.IDs = IDs
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.feature_data = data[features]
        self.target_data = data[target]
    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        i = self.IDs[idx]

        # Extract feature sequence
        X = self.feature_data.loc[i:i + self.sequence_length].values.astype(dtype='float32')

        # Extract target(s)
        y = self.target_data.loc[i + self.sequence_length + self.horizon - 1].values.astype(dtype='float32')

        return torch.from_numpy(X), torch.from_numpy(y), i
