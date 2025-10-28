import lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.classification import ConfusionMatrix, MulticlassAccuracy


class T2VTransformerClassifier(nn.Module):
    def __init__(
        self,
        time_dim,
        n_features,
        output_size,
        kernel_size=1,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        pool="last",
    ):  # "last" | "mean"
        super().__init__()
        self.time_dim = time_dim
        self.n_features = n_features
        self.pool = pool
        # Time2Vec embedding
        self.time2vec = Time2Vec(time_dim, kernel_size)

        self.positional_encoding = PositionalEncoding(d_model, dropout)
        # Project Time2Vec output to d_model

        self.input_proj = nn.Linear(time_dim * (kernel_size + 1) + n_features, d_model)
        # Project Time2Vec output to d_model

        if self.pool == "attn":
            self.pooler = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Tanh(),
                nn.Linear(d_model, 1),
            )

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
        X_time = x[:, :, : self.time_dim]  # (B, T, time_dim)
        X_other = x[:, :, self.time_dim :]  # (B, T, feature_dim)
        X_time = self.time2vec(X_time)  # (B, T, time_dim + time_dim * k)
        x = torch.cat(
            [X_time, X_other], dim=-1
        )  # (B, T, feature_dim + time_dim * (1+k)
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.positional_encoding(x)  # (B, T, d_model)
        x = self.encoder(x)  # (B, T, d_model)
        if self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "attn":
            a = torch.softmax(self.pooler(x), dim=1)
            x = torch.sum(a * x, dim=1)
        else:  # last
            x = x[:, -1, :]

        # Output classification logits
        logits = self.fc_out(x)  # (B, output_size)
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


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(
            torch.empty(max_len, 1, d_model)
        )  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class T2VTransformerClassifierModule(pl.LightningModule):
    def __init__(
        self,
        n_timefeatures=4,
        n_features=1,
        output_size=3,
        kernel_size=1,
        d_model=64,
        nhead=4,
        n_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        label_smoothing=0.0,
        pool="mean",
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = T2VTransformerClassifier(
            time_dim=n_timefeatures,
            n_features=n_features,
            output_size=output_size,
            kernel_size=kernel_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pool=pool,
        )
        self.output_size = output_size
        self.val_preds = []
        self.val_labels = []

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.val_acc = MulticlassAccuracy(num_classes=output_size)
        self.test_acc = MulticlassAccuracy(num_classes=output_size)
        self.lr = lr

    def forward(self, x):
        return self.model(x)  # logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze().long()
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze().long()
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        # Store predictions and labels for confusion matrix
        self.val_preds.append(preds)
        self.val_labels.append(y)
        return loss

    def on_validation_epoch_end(self):
        val_preds = torch.cat(self.val_preds)
        val_labels = torch.cat(self.val_labels)

        conf_mat = ConfusionMatrix(task="multiclass", num_classes=self.output_size)
        # Compute confusion matrix
        cm = conf_mat(val_preds.cpu(), val_labels.cpu())

        # Plot confusion matrix
        self.plot_confusion_matrix(cm)

        # Clear stored predictions and labels for the next epoch
        self.val_preds.clear()
        self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze().long()
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, y)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def plot_confusion_matrix(self, cm):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix")

        # Log confusion matrix to TensorBoard
        self.logger.experiment.add_figure("Confusion Matrix", fig, self.current_epoch)
        plt.close(fig)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        # OneCycleLR (optional): replace StepLR for smoother training
        # steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        # sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=self.trainer.estimated_stepping_batches)
        # return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        return [opt], [sched]
