import lightning as L
import torch
from torch import nn


class ForexGRU(L.LightningModule):
    def __init__(self, horizon: int):
        super().__init__()
        self.save_hyperparameters()

        self.gru = nn.GRU(
            input_size=1,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(128, self.hparams.horizon)
        self.loss_fn = nn.HuberLoss()  # Use BCEWithLogitsLoss() for binary classification

    def forward(self, x):
        out, _ = self.gru(x)
        last_out = out[:, -1, :]  # get output at last time step
        # return torch.tanh(self.fc(last_out))
        return self.fc(last_out)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).squeeze(-1)
        loss = self.loss_fn(preds, y.squeeze(-1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).squeeze(-1)
        loss = self.loss_fn(preds, y.squeeze(-1))
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def get_progress_bar_dict(self):
        """Customize metrics shown in the progress bar."""
        items = super().get_progress_bar_dict()
        # Format float values to 6 digits
        for k, v in items.items():
            if isinstance(v, float):
                items[k] = f"{v:.6f}"
        return items
