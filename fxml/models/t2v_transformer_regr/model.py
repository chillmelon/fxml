import torch
from torch import nn

from fxml.models.base_regressor import BaseRegressorModule
from fxml.models.transformer_regressor.model import PositionalEncoding


class T2VTransformerRegressor(nn.Module):
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
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        self.input_proj = nn.Linear(time_dim * (kernel_size + 1) + n_features, d_model)
        # Project Time2Vec output to d_model

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


class T2VTransformerRegressorModule(BaseRegressorModule):
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
        label_smoothing=0.0,  # Kept for backward compatibility (unused)
        pool="mean",
        lr=1e-3,
        optimizer_type="Adam",
        weight_decay=1e-4,
        scheduler_step_size=10,
        scheduler_gamma=0.5,
        enable_plotting=True,
    ):
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
        self.model = T2VTransformerRegressor(
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
