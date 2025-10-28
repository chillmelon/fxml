from torch import nn

from fxml.models.base_classifier import BaseClassifierModule


class BaselineClassifier(nn.Module):
    """Simple baseline classifier using mean pooling and feedforward layers.

    This model serves as a baseline by:
    1. Mean pooling across the sequence dimension
    2. Passing through a simple feedforward network

    This provides a non-sequential baseline to compare against LSTM/Transformer models.
    """

    def __init__(
        self,
        n_features,
        output_size,
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
        logits = self.fc_out(x)  # (B, output_size)

        return logits


class BaselineClassifierModule(BaseClassifierModule):
    """
    Baseline classifier Lightning module.

    Uses a simple non-sequential baseline model with mean pooling
    across the time dimension followed by feedforward layers.
    """

    def __init__(
        self,
        n_features=1,
        output_size=3,
        n_hidden=64,
        dropout=0.1,
        label_smoothing=0.0,
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

        # Create the baseline model architecture
        self.model = BaselineClassifier(
            n_features=n_features,
            output_size=output_size,
            n_hidden=n_hidden,
            dropout=dropout,
        )
