import torch
from torch import nn
import lightning as pl
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

class GRUModel(nn.Module):
    def __init__(self, n_features, n_hidden, n_layers, dropout, output_size):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(n_hidden, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        last_out = out[:, -1, :]              # <== use output at last time step
        last_out = self.dropout(last_out)
        return self.regressor(last_out)


class GRURegressorModule(pl.LightningModule):
    def __init__(self, n_features=1, horizon=1, n_hidden=64, n_layers=2, dropout=0.0, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = GRUModel(
            n_features=self.hparams.n_features,
            n_hidden=self.hparams.n_hidden,
            n_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout,
            output_size=self.hparams.horizon
        )

        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        output = self.model(x)

        loss = self.criterion(output, y)
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
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, out = self(x, y)
        self.log('test_loss', loss, prog_bar=True)
        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
