import torch
from torch import nn

from fxml.models.base_classifier import BaseClassifierModule


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
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x):  # x: (B, T, F)
        x = self.input_proj(x)

        if self.pool == "cls":
            B = x.size(0)
            cls = self.cls.expand(B, 1, -1)  # (B,1,D)
            x = torch.cat([cls, x], dim=1)  # prepend CLS

        x = self.positional_encoding(x)
        x = self.encoder(x)

        if self.pool == "cls":
            out = x[:, 0]
        elif self.pool == "mean":
            out = x.mean(dim=1)
        else:
            out = x[:, -1]
        return self.fc_out(out)


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


class TransformerClassifierModule(BaseClassifierModule):
    """
    Transformer classifier Lightning module.

    Uses a Transformer encoder with self-attention for sequence modeling,
    with optional mean or last pooling strategy.
    """

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
        lr=1e-3,
        optimizer="adamw",
        weight_decay=1e-4,
    ):
        super().__init__(
            n_features=n_features,
            output_size=output_size,
            lr=lr,
            label_smoothing=label_smoothing,
            optimizer=optimizer,
            weight_decay=weight_decay,
        )

        # Create the transformer model architecture
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
