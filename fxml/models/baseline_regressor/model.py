import torch
from torch import nn

from fxml.models.base_regressor import BaseRegressorModule


class BaselineRegressor(nn.Module):
    """Simple baseline regressor using mean pooling and feedforward layers.

    This model serves as a baseline by:
    1. Mean pooling across the sequence dimension
    2. Passing through a simple feedforward network

    This provides a non-sequential baseline to compare against LSTM/Transformer models.
    """

    def __init__(
        self,
        n_features,
        output_size=1,
        n_hidden=64,
        dropout=0.1,
    ):
        super().__init__()

        # Simple feedforward network after mean pooling
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(n_hidden, n_hidden // 2)
        self.fc_out = nn.Linear(n_hidden // 2, output_size)

    def forward(self, x):
        # x: (B, T, n_features)
        # Mean pooling across time dimension
        x = x.mean(dim=1)  # (B, n_features)

        # Feedforward network
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        preds = self.fc_out(x)  # (B, output_size)

        return preds


class BaselineRegressorModule(BaseRegressorModule):
    """Baseline regressor Lightning module.

    Uses mean pooling + feedforward network for regression.
    """

    def __init__(
        self,
        n_features=1,
        output_size=1,
        n_hidden=64,
        dropout=0.1,
        lr=1e-3,
        optimizer_type="AdamW",
        weight_decay=1e-4,
        scheduler_step_size=10,
        scheduler_gamma=0.5,
        enable_plotting=True,
    ):
        """Initialize baseline regressor module.

        Args:
            n_features: Number of input features
            output_size: Number of output dimensions
            n_hidden: Hidden layer size
            dropout: Dropout probability
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
        self.model = BaselineRegressor(
            n_features=n_features,
            output_size=output_size,
            n_hidden=n_hidden,
            dropout=dropout,
        )
