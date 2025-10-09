import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DirectionDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        labels: pd.DataFrame,
        sequence_length: int,
        features_cols: list,
        target_col: str,
    ):
        """
        data: DataFrame with indexed datetime and full market data (must contain all feature and target columns)
        events: DataFrame with event index (timestamps) and target column
        sequence_length: number of timesteps before each event to include
        features_cols: list of feature column names
        target_col: name of the target column (must be in events)
        """
        self.raw_data = data
        self.labels = labels
        self.sequence_length = sequence_length
        self.features_cols = features_cols
        self.target_col = target_col
        self.features = data[features_cols]

        self.t_events = labels.index
        self.X, self.y, self.t_events = self._create_sequences()

    def _create_sequences(self):
        print("----- Start Creating Sequences -----")
        X, y, valid_t_events = [], [], []
        for event_time in self.t_events:
            try:
                end_idx: int = int(self.raw_data.index.searchsorted(event_time))
            except (KeyError, TypeError):
                continue  # Skip if event_time not in index or invalid

            start_idx = end_idx - self.sequence_length
            if start_idx < 0:
                continue  # Not enough history

            seq = self.features.iloc[start_idx:end_idx].values

            label = self.labels.loc[event_time, self.target_col].values

            if len(seq) == self.sequence_length:
                X.append(seq)
                y.append(label)
                valid_t_events.append(event_time)  # Keep only used event times

        print(f"----- End Creating {len(valid_t_events)} Sequences -----")
        return (
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float64),
            pd.Index(valid_t_events),
        )

    def __len__(self):
        return len(self.t_events)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),
            torch.tensor(self.y[i], dtype=torch.long),
            i,
        )
