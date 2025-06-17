import lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch._prims_common import Dim
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
)


class T2VTransformerModel(nn.Module):
    def __init__(
        self,
        time_dim,
        feature_dim,
        output_size,
        d_model,
        nhead,
        dim_feedforward,
        num_layers,
        kernel_size=1,
        dropout=0.1,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.feature_dim = feature_dim
        self.output_size = output_size
        self.d_model = d_model

        # Time2Vec embedding
        self.time2vec = Time2Vec(time_dim, kernel_size)

        # Project Time2Vec output to d_model
        time2vec_output_dim = time_dim + time_dim * kernel_size

        self.input_proj = nn.Linear(time2vec_output_dim + feature_dim, d_model)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, X):
        X_time = X[:, :, : self.time_dim]  # (B, T, time_dim)
        X_other = X[:, :, self.time_dim :]  # (B, T, feature_dim)
        X_time = self.time2vec(X_time)  # (B, T, input_dim + input_dim * k)
        X = torch.cat(
            [X_time, X_other], dim=-1
        )  # (B, T, feature_dim + time_dim * (1+k)
        X = self.input_proj(X)  # (B, T, d_model)
        X = self.encoder(X)
        X = X[:, -1, :]  # (B, d_model)

        # Output classification logits
        logits = self.fc_out(X)  # (B, output_size)
        return logits


class Time2Vec(nn.Module):
    def __init__(self, input_dim: int, kernel_size: int = 1):
        super(Time2Vec, self).__init__()
        self.input_dim = input_dim
        self.k = kernel_size

        # Linear term per feature
        self.wb = nn.Parameter(torch.empty(1, 1, input_dim))
        self.bb = nn.Parameter(torch.empty(1, 1, input_dim))
        nn.init.xavier_uniform_(self.wb)
        nn.init.zeros_(self.bb)

        # Periodic terms per feature
        self.wa = nn.Parameter(torch.rand(1, input_dim, kernel_size))
        self.ba = nn.Parameter(torch.rand(1, input_dim, kernel_size))

    def forward(self, x):
        # x: (B, T, input_dim)
        trend = self.wb * x + self.bb  # (B, T, input_dim)

        # For periodic, we want: (B, T, input_dim, k)
        x_exp = x.unsqueeze(-1)  # (B, T, input_dim, 1)
        wa = self.wa  # (1, input_dim, k)
        ba = self.ba  # (1, input_dim, k)

        periodic = torch.sin(x_exp * wa + ba)  # (B, T, input_dim, k)
        periodic = periodic.contiguous().reshape(x.shape[0], x.shape[1], -1)
        out = torch.cat([trend, periodic], dim=-1)  # (B, T, input_dim + input_dim * k)
        return out


class T2VTransformerModule(pl.LightningModule):
    def __init__(
        self,
        n_time=1,
        n_features=1,
        output_size=3,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        kernel_size=1,
        dropout=0.1,
        label_smoothing=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = T2VTransformerModel(
            time_dim=self.hparams.n_time,
            feature_dim=self.hparams.n_features,
            output_size=self.hparams.output_size,
            d_model=self.hparams.d_model,
            nhead=self.hparams.nhead,
            num_layers=self.hparams.num_layers,
            dim_feedforward=self.hparams.dim_feedforward,
            kernel_size=self.hparams.kernel_size,
            dropout=self.hparams.dropout,
        )

        # metrics
        self.val_acc = MulticlassAccuracy(num_classes=self.hparams.output_size)
        self.val_precision = MulticlassPrecision(num_classes=self.hparams.output_size)
        self.val_recall = MulticlassRecall(num_classes=self.hparams.output_size)
        self.test_acc = MulticlassAccuracy(num_classes=self.hparams.output_size)
        self.test_precision = MulticlassPrecision(num_classes=self.hparams.output_size)
        self.test_recall = MulticlassRecall(num_classes=self.hparams.output_size)

    def forward(self, x, labels=None):
        return self.model(x)

    def compute_loss(self, logits, targets):
        class_weights = self.trainer.datamodule.class_weights.to(self.device)
        return F.cross_entropy(
            logits,
            targets,
            weight=class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        y = y.squeeze().long()
        loss = self.compute_loss(logits, y)

        self.log("train_loss", loss, prog_bar=True, logger=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)

        # Always use unweighted loss for validation
        val_loss = F.cross_entropy(logits, y)  # No class weights
        self.log("val_loss", val_loss, prog_bar=True)

        preds = torch.argmax(logits, dim=1)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_acc(preds, y)

        self.log("val_precision", self.val_precision, prog_bar=True)
        self.log("val_recall", self.val_recall, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return {"loss": val_loss}

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)

        # Always use unweighted loss for validation
        test_loss = F.cross_entropy(logits, y)  # No class weights
        self.log("test_loss", test_loss, prog_bar=True)

        preds = torch.argmax(logits, dim=1)
        self.test_precision(preds, y)
        self.test_recall(preds, y)
        self.test_acc(preds, y)

        self.log("test_precision", self.val_precision, prog_bar=True)
        self.log("test_recall", self.val_recall, prog_bar=True)
        self.log("test_acc", self.val_acc, prog_bar=True)
        return {"loss": test_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]
