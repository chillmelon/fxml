from collections import Counter

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset.direction_dataset import DirectionDataset


class EventBasedDataModule(L.LightningDataModule):
    def __init__(
        self,
        data,
        labels,
        sequence_length: int,
        features: list,
        target: str,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split: float = 0.2,
        shuffle: bool = True,
        random_state: int = 42,
        balanced_sampling: bool = True,  # <<< 這個開關
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["data", "labels"])
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length
        self.features = features
        self.target = target
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.shuffle = shuffle
        self.random_state = random_state
        self.balanced_sampling = balanced_sampling

        self.class_weights = None
        self._train_sample_weights = None

    def setup(self, stage=None):
        print("====== Start Setting Up Data Module =====")
        # sort by time index to maintain temporal order
        sorted_events = self.labels.sort_index()
        n_val = int(len(sorted_events) * self.val_split)

        val_events = sorted_events.iloc[-n_val:]
        train_events = sorted_events.iloc[:-n_val]

        print("====== Start Building Training Dataset =====")
        self.train_dataset = DirectionDataset(
            data=self.data,
            events=train_events,
            sequence_length=self.sequence_length,
            features_cols=self.features,
            target_col=self.target,
        )

        print("====== Start Building Validation Dataset =====")
        self.val_dataset = DirectionDataset(
            data=self.data,
            events=val_events,
            sequence_length=self.sequence_length,
            features_cols=self.features,
            target_col=self.target,
        )

        print("====== Start Calculating Class/Sample Weights =====")
        # 取得所有訓練標籤（DirectionDataset 需有 .y）
        y = np.asarray(self.train_dataset.y, dtype=np.int64)
        # 類別數；若你的 label 是 {0,1,2} 以外，請改成固定 minlength=output_size
        num_classes = int(y.max()) + 1
        counts = np.bincount(y, minlength=num_classes)  # [n0, n1, n2, ...]
        N, C = counts.sum(), len(counts)

        # (a) CrossEntropy 用的類別權重（與 1/n 成比例）
        class_weights = torch.tensor(N / (C * counts), dtype=torch.float32)
        self.class_weights = class_weights  # LightningModule 會在 on_fit_start 拿

        # (b) Sampler 用的樣本權重（每筆）
        if self.balanced_sampling:
            invfreq = 1.0 / counts
            sample_weights = invfreq[y]  # 長度 = len(train_dataset)
            # WeightedRandomSampler 需要 torch.DoubleTensor
            self._train_sample_weights = torch.as_tensor(
                sample_weights, dtype=torch.double
            )
        print(f"Counts={counts.tolist()}, class_weights={self.class_weights.tolist()}")
        print("====== End Setting Up Data Module =====")

    def train_dataloader(self):
        if self.balanced_sampling:
            sampler = WeightedRandomSampler(
                weights=self._train_sample_weights,
                num_samples=len(
                    self._train_sample_weights
                ),  # 每 epoch 看一遍（近似均衡）
                replacement=True,
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,  # <<< 使用 sampler
                shuffle=False,  # <<< sampler 下請勿 shuffle
                num_workers=self.num_workers,
                pin_memory=(self.num_workers > 0),
                persistent_workers=(self.num_workers > 0),
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=(self.num_workers > 0),
                persistent_workers=(self.num_workers > 0),
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.num_workers > 0),
            persistent_workers=(self.num_workers > 0),
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.num_workers > 0),
        )
