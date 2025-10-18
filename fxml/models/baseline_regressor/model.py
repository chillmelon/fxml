import lightning as pl
import torch
from torch import nn


class BaselineRegressor(nn.Module):
    """Simple baseline regressor using mean pooling and feedforward layers.

    This model serves as a baseline by:
    1. Mean pooling across the sequence dimension
    2. Passing through a simple feedforward network

    This provides a non-sequential baseline to compare against LSTM/Transformer models.
    """

    def __init__(
        self,
        n_features,
        output_size=1,
        n_hidden=64,
        dropout=0.1,
    ):
        super().__init__()

        # Simple feedforward network after mean pooling
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(n_hidden, n_hidden // 2)
        self.fc_out = nn.Linear(n_hidden // 2, output_size)

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
        preds = self.fc_out(x)  # (B, output_size)

        return preds


class BaselineRegressorModule(pl.LightningModule):
    def __init__(
        self,
        n_features=1,
        output_size=1,
        n_hidden=64,
        dropout=0.1,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = BaselineRegressor(
            n_features=n_features,
            output_size=output_size,
            n_hidden=n_hidden,
            dropout=dropout,
        )

        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, x, labels=None):
        pred = self.model(x)
        loss = 0
        if labels is not None:
            labels = labels.view(-1, 1)
            loss = self.criterion(pred, labels)
        return loss, pred

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        loss, out = self(x, y)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        loss, out = self(x, y)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        loss, out = self(x, y)

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        return [opt], [sched]
