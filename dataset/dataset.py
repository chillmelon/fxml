import torch
from torch.utils.data import Dataset
import numpy as np

class ForexDataset(Dataset):
    def __init__(self, data, sequence_length, features, target, stride=1):
        self.data = data.reset_index(drop=True)
        self.sequence_length = sequence_length
        self.features = features
        self.target = target
        self.stride = stride

        # Convert features/labels to NumPy for fast slicing
        self.feature_data = data[features].to_numpy(dtype=np.float32)
        self.target_data = data[target].to_numpy(dtype=np.int64)

        # Precompute valid sequence start indices
        self.indices = self._generate_indices()
        print('done initializing dataset')

    def _generate_indices(self):
        # Sequences must have sequence_length + 1 (for target)
        max_start = len(self.data) - self.sequence_length - 1
        return list(range(0, max_start + 1, self.stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        X = self.feature_data[idx : idx + self.sequence_length]
        y = self.target_data[idx + self.sequence_length]  # classification target
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long), idx
