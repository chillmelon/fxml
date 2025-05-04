import torch
from torch import nn
import lightning as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

class LSTMRegressor(nn.Module):
    def __init__(self, n_features, output_size, n_hidden, n_layers, dropout):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        # self.batch_norm = nn.BatchNorm1d(n_hidden)
        self.regressor = nn.Linear(n_hidden, output_size)


    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        last_hidden = self.dropout(last_hidden)
        # normalized = self.batch_norm(last_hidden)
        return self.regressor(last_hidden)


class LSTMRegressorModule(pl.LightningModule):
    def __init__(self, n_features=1, horizon=1, n_hidden=64, n_layers=2, dropout=0.0, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = LSTMRegressor(
            n_features=self.hparams.n_features,
            output_size=self.hparams.horizon,
            n_hidden=self.hparams.n_hidden,
            n_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout,
        )

        self.criterion = nn.MSELoss()


    def forward(self, x, labels=None):
        output = self.model(x)
        loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, out = self(x, y)

        self.log('train_loss', loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, out = self(x, y)
        self.log('val_loss', loss, prog_bar=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, out = self(x, y)
        self.log('test_loss', loss, prog_bar=True)
        return {"loss": loss}

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=1e-4)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
