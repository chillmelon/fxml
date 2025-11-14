import torch
from torch import nn

from fxml.models.base_regressor import BaseRegressorModule
from fxml.models.t2v_transformer_regr.model import Time2Vec


class T2VLSTMRegressor(nn.Module):
    """LSTM-based regressor for sequential data."""

    def __init__(
        self,
        n_features,
        output_size,
        kernel_size=1,
        n_hidden=64,
        n_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.time_dim = n_features
        self.n_features = n_features
        # Time2Vec embedding
        self.time2vec = Time2Vec(n_features, kernel_size)

        input_size = n_features + 2 * n_features * kernel_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.linear = nn.Linear(n_hidden, output_size)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = self.time2vec(x)  # (B, T, time_dim + time_dim * k)
        _, (hidden, _) = self.lstm(x)
        preds = self.linear(hidden[-1])
        return preds


class T2VLSTMRegressorModule(BaseRegressorModule):
    """LSTM regressor Lightning module."""

    def __init__(
        self,
        n_features=1,
        output_size=1,
        kernel_size=1,
        n_hidden=64,
        n_layers=2,
        dropout=0.0,
        lr=1e-2,
        optimizer_type="Adam",
        weight_decay=0.0,
        scheduler_step_size=10,
        scheduler_gamma=0.1,
        enable_plotting=True,
    ):
        """Initialize LSTM regressor module.

        Args:
            n_features: Number of input features
            output_size: Number of output dimensions
            n_hidden: LSTM hidden size
            n_layers: Number of LSTM layers
            dropout: Dropout probability (applied between LSTM layers)
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
        self.model = T2VLSTMRegressor(
            n_features=n_features,
            output_size=output_size,
            kernel_size=kernel_size,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout=dropout,
        )
