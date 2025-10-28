import torch
from torch import nn

from fxml.models.base_regressor import BaseRegressorModule


class GRURegressorModel(nn.Module):
    """GRU-based regressor for sequential data."""

    def __init__(self, n_features, output_size, n_hidden, n_layers, dropout):
        super().__init__()

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.linear = nn.Linear(n_hidden, output_size)

    def forward(self, x):
        self.gru.flatten_parameters()
        _, hidden = self.gru(x)
        return self.linear(hidden[-1])


class GRURegressorModule(BaseRegressorModule):
    """GRU regressor Lightning module."""

    def __init__(
        self,
        n_features=1,
        output_size=1,
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
        """Initialize GRU regressor module.

        Args:
            n_features: Number of input features
            output_size: Number of output dimensions
            n_hidden: GRU hidden size
            n_layers: Number of GRU layers
            dropout: Dropout probability (applied between GRU layers)
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
        self.model = GRURegressorModel(
            n_features=n_features,
            output_size=output_size,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout=dropout,
        )
