import lightning as pl
import torch
from torch import nn


class GRUModel(nn.Module):
    def __init__(self, n_features, output_size, n_hidden, n_layers, dropout):
        super().__init__()

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.linear = nn.Linear(n_hidden, output_size)

    def forward(self, x):
        self.gru.flatten_parameters()
        _, hidden = self.gru(x)
        return self.linear(hidden[-1])


class GRUModule(pl.LightningModule):
    def __init__(
        self, n_features=1, output_size=1, n_hidden=64, n_layers=2, dropout=0.0
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = GRUModel(
            n_features=self.hparams.n_features,
            output_size=self.hparams.output_size,
            n_hidden=self.hparams.n_hidden,
            n_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout,
        )

        self.criterion = nn.MSELoss()

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
