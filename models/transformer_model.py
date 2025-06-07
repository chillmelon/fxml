import torch
from torch import nn
import lightning as pl
from torch._prims_common import Dim
from torchmetrics.classification import MulticlassAccuracy


class TransformerModel(nn.Module):
    def __init__(self, n_features, output_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(n_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # Use the final time step
        logits = self.fc_out(x)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModule(pl.LightningModule):
    def __init__(self, n_features=1, output_size=3, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.save_hyperparameters()

        self.model = TransformerModel(
            n_features=self.hparams.n_features,
            output_size=self.hparams.output_size,
            d_model=self.hparams.d_model,
            nhead=self.hparams.nhead,
            num_layers=self.hparams.num_layers,
            dim_feedforward=self.hparams.dim_feedforward,
            dropout=self.hparams.dropout
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            labels = labels.view(-1).long()
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        loss, out = self(x, y)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        loss, out = self(x, y)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        loss, out = self(x, y)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]
