import torch
from torch.utils.data import Dataset

class ForexClassificationDataset(Dataset):
    """Dataset for sequence classification with multi-step horizon."""

    def __init__(self, data, sequence_length, horizon, stride, features, target, group_col='time_group'):
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.features = features
        self.target = target
        self.stride = stride
        self.group_col = group_col

        # Reset index to ensure integer indexing is valid
        self.data = data.reset_index(drop=True)
        self.feature_data = self.data[self.features].values
        self.target_data = self.data[self.target].values
        self.group_labels = self.data[self.group_col].values

        self.IDs = self._get_valid_sequence_starts()

    def _get_valid_sequence_starts(self):
        indices = []
        group_indices = {}

        for idx, group in enumerate(self.group_labels):
            group_indices.setdefault(group, []).append(idx)

        for group, idxs in group_indices.items():
            idxs = sorted(idxs)
            max_start = len(idxs) - (self.sequence_length + self.horizon) + 1
            for start in range(0, max_start, self.stride):
                indices.append(idxs[start])

        return indices

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        start = idx
        end = idx + self.sequence_length
        target_idx = start + self.sequence_length + self.horizon - 1

        x = torch.tensor(self.feature_data[start:end], dtype=torch.float32)
        y = torch.tensor(self.target_data[target_idx], dtype=torch.long)

        return x, y
