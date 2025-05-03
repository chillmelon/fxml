from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import numpy as np

class Splitter:
    def __init__(
        self,
        df,
        IDs,
        sequence_length,
        horizon,
        target_col=None,
        method=None,
        test_size=0.2,
        random_state=42
    ):
        """
        Args:
            df (pd.DataFrame): Full dataset
            IDs (list or array): Valid sequence start indices
            sequence_length (int): Input sequence length
            horizon (int): Steps to predict ahead
            target_col (str): Target column name (required for 'stratified')
            method (str): 'random', 'stratified', or None for ordered split
            test_size (float): Fraction for validation split
            random_state (int): Seed for reproducibility
        """
        assert method in [None, 'random', 'stratified'], "Invalid split method"

        self.df = df
        self.IDs = np.array(IDs)
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.target_col = target_col
        self.method = method
        self.test_size = test_size
        self.random_state = random_state

    def split(self):
        if self.method == 'random':
            return self._random_split()
        elif self.method == 'stratified':
            assert self.target_col is not None, "target_col is required for stratified split"
            return self._stratified_split()
        else:
            return self._normal_split()

    def _normal_split(self):
        """Time-ordered split (no shuffle)."""
        train_idx, val_idx = train_test_split(
            self.IDs,
            test_size=self.test_size,
            shuffle=False
        )
        return train_idx, val_idx

    def _random_split(self):
        """Random shuffled split."""
        train_idx, val_idx = train_test_split(
            self.IDs,
            test_size=self.test_size,
            shuffle=True,
            random_state=self.random_state
        )
        return train_idx, val_idx

    def _stratified_split(self):
        """Stratified split by target label at (start + seq_len + horizon - 1)."""
        target_indices = self.IDs + self.sequence_length + self.horizon - 1
        labels = self.df.loc[target_indices, self.target_col].values

        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.random_state
        )
        train_idx, val_idx = next(splitter.split(self.IDs, labels))

        return train_idx.tolist(), val_idx.tolist()
