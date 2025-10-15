import lightning as pl
import torch
from torch import nn


class LSTMRegressor(nn.Module):
    def __init__(self, n_features, output_size, n_hidden, n_layers, dropout):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.linear = nn.Linear(n_hidden, output_size)

    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        preds = self.linear(last_hidden)
        return preds


class LSTMRegressorModule(pl.LightningModule):
    def __init__(
        self, n_features=1, output_size=1, n_hidden=64, n_layers=2, dropout=0.0
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = LSTMRegressor(
            n_features=n_features,
            output_size=output_size,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.criterion = nn.MSELoss()

    def forward(self, x, targets=None):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        outs = self(x)
        loss = self.criterion(outs, y)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        outs = self(x)
        loss = self.criterion(outs, y)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        loss, out = self(x, y)

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
