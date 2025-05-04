import torch
from torch import nn
import lightning as pl
from torchmetrics.classification import MulticlassAccuracy

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class TransformerModel(nn.Module):
    def __init__(self, n_features, d_model, n_heads, num_layers, n_classes, dropout):
        super().__init__()

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.input_proj(x)              # (batch, seq_len, d_model)
        x = self.pos_encoder(x)             # add positional encoding
        x = self.transformer(x)             # (batch, seq_len, d_model)
        x = x[:, -1, :]                     # use the last token representation
        return self.classifier(x)           # (batch, n_classes)

class TransformerClassifierModule(pl.LightningModule):
    def __init__(
        self,
        n_features=1,
        n_classes=3,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        lr=1e-4
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = TransformerModel(
            n_features=self.hparams.n_features,
            d_model=self.hparams.d_model,
            n_heads=self.hparams.n_heads,
            num_layers=self.hparams.n_layers,
            n_classes=self.hparams.n_classes,
            dropout=self.hparams.dropout
        )

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=n_classes)
        self.val_acc = MulticlassAccuracy(num_classes=n_classes)
        self.test_acc = MulticlassAccuracy(num_classes=n_classes)

    def forward(self, x, labels=None):
        out = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(out, labels)
        return loss, out

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, out = self(x, y)
        preds = torch.argmax(out, dim=1)
        acc = self.train_acc(preds, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, out = self(x, y)
        preds = torch.argmax(out, dim=1)
        acc = self.val_acc(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, out = self(x, y)
        preds = torch.argmax(out, dim=1)
        acc = self.test_acc(preds, y)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
