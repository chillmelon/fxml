import torch
from torch import nn

from fxml.models.base_regressor import BaseRegressorModule


class TransformerRegressor(nn.Module):
    """Transformer-based regressor with CLS token for sequence pooling."""

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
    """Sinusoidal positional encoding for Transformer models."""

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


class TransformerRegressorModule(BaseRegressorModule):
    """Transformer regressor Lightning module."""

    def __init__(
        self,
        n_features=1,
        output_size=1,
        d_model=64,
        nhead=4,
        n_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        label_smoothing=0.0,  # Kept for backward compatibility (unused)
        pool="mean",
        lr=3e-4,
        optimizer_type="Adam",
        weight_decay=1e-4,
        scheduler_step_size=10,
        scheduler_gamma=0.5,
        enable_plotting=True,
    ):
        """Initialize Transformer regressor module.

        Args:
            n_features: Number of input features
            output_size: Number of output dimensions
            d_model: Transformer model dimension
            nhead: Number of attention heads
            n_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            label_smoothing: Kept for backward compatibility (unused)
            pool: Pooling strategy ("last" or "mean")
            lr: Learning rate
            optimizer_type: Type of optimizer ("Adam" or "AdamW")
            weight_decay: Weight decay for regularization
            scheduler_step_size: Step size for learning rate scheduler
            scheduler_gamma: Multiplicative factor for learning rate decay
            enable_plotting: Whether to enable validation plotting
        """
        super().__init__(
            lr=lr,
            optimizer_type=optimizer_type,
            weight_decay=weight_decay,
            scheduler_step_size=scheduler_step_size,
            scheduler_gamma=scheduler_gamma,
            enable_plotting=enable_plotting,
        )
        self.save_hyperparameters()

        # Build model
        self.model = TransformerRegressor(
            n_features=n_features,
            output_size=output_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pool=pool,
        )
