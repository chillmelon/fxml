import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class ForexClassificationDataset(Dataset):
    """Dataset for forex price data with sequence-based inputs."""

    def __init__(self, data, sequence_length, features, target, transform=None):
        """
        Initialize the dataset.

        Args:
            data (pandas.DataFrame): The forex data.
            sequence_length (int): The length of the input sequence.
            features (list): The feature columns to use.
            target (str): The target column to predict.
            transform (callable, optional): Optional transform to be applied to samples.
        """
        self.data = data
        self.sequence_length = sequence_length
        self.features = features
        self.target = target

        # Prepare data
        self._prepare_data()

    def _prepare_data(self):
        """Prepare the data for training."""
        # Extract features and target
        features_data = self.data[self.features].values
        target_data = self.data[self.target].values

        # Create sequences
        X, y = [], []
        for i in range(len(features_data) - self.sequence_length):
            X.append(features_data[i:i+self.sequence_length])
            # For multi-step forecasting, get a sequence of future values
            y.append(target_data[i + self.sequence_length])

        # Convert to tensors
        self.X = torch.FloatTensor(np.array(X))
        self.y = torch.LongTensor(np.array(y))

    def __len__(self):
        """Return the number of samples."""
        return len(self.X)

    def __getitem__(self, idx):
        """Return a sample."""
        return self.X[idx], self.y[idx]
