import lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.classification import MulticlassAccuracy


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        n_features,
        output_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        pool="last",
    ):  # "last" | "mean"
        super().__init__()
        self.pool = pool
        self.input_proj = nn.Linear(n_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x):
        # x: (B, T, n_features)
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)  # (B, T, d_model)
        if self.pool == "mean":
            x = x.mean(dim=1)
        else:
            x = x[:, -1, :]
        logits = self.fc_out(x)  # (B, C)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=8192):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


class TransformerClassifierModule(pl.LightningModule):
    def __init__(
        self,
        n_features=1,
        output_size=3,
        d_model=64,
        nhead=4,
        n_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        label_smoothing=0.0,
        pool="mean",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerClassifier(
            n_features=n_features,
            output_size=output_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pool=pool,
        )
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
        )
        self.train_acc = MulticlassAccuracy(num_classes=output_size)
        self.val_acc = MulticlassAccuracy(num_classes=output_size)

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
        opt = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-4)
        # OneCycleLR (optional): replace StepLR for smoother training
        # steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        # sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=self.trainer.estimated_stepping_batches)
        # return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        return [opt], [sched]
