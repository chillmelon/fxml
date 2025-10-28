import lightning as pl
import torch
from torch import nn


class LSTMClassifier(nn.Module):
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
        logits = self.linear(last_hidden)
        return logits


class LSTMClassifierModule(pl.LightningModule):
    def __init__(
        self, n_features=1, output_size=3, n_hidden=64, n_layers=2, dropout=0.0, lr=1e-3
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = LSTMClassifier(
            n_features=n_features,
            output_size=output_size,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        return self.model(x)  # logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze().long()
        logits = self(x)
        loss = self.criterion(logits, y)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze().long()
        logits = self(x)
        loss = self.criterion(logits, y)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze().long()
        logits = self(x)
        loss = self.criterion(logits, y)

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]
