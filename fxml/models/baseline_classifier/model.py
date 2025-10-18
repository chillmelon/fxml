import lightning as pl
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy


class BaselineClassifier(nn.Module):
    """Simple baseline classifier using mean pooling and feedforward layers.

    This model serves as a baseline by:
    1. Mean pooling across the sequence dimension
    2. Passing through a simple feedforward network

    This provides a non-sequential baseline to compare against LSTM/Transformer models.
    """

    def __init__(
        self,
        n_features,
        output_size,
        hidden_size=64,
        dropout=0.1,
    ):
        super().__init__()

        # Simple feedforward network after mean pooling
        self.fc1 = nn.Linear(n_features, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc_out = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        # x: (B, T, n_features)
        # Mean pooling across time dimension
        x = x.mean(dim=1)  # (B, n_features)

        # Feedforward network
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc_out(x)  # (B, output_size)

        return logits


class BaselineClassifierModule(pl.LightningModule):
    def __init__(
        self,
        n_features=1,
        output_size=3,
        hidden_size=64,
        dropout=0.1,
        label_smoothing=0.0,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = BaselineClassifier(
            n_features=n_features,
            output_size=output_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.train_acc = MulticlassAccuracy(num_classes=output_size)
        self.val_acc = MulticlassAccuracy(num_classes=output_size)
        self.lr = lr

    def forward(self, x):
        return self.model(x)  # logits

    def _step(self, batch, stage):
        x, y, _ = batch
        y = y.squeeze().long()
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        if stage == "train":
            acc = self.train_acc(preds, y)
            self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        elif stage == "val":
            acc = self.val_acc(preds, y)
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        else:
            self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        return [opt], [sched]
