from torch import nn

from fxml.models.base_classifier import BaseClassifierModule


class LSTMClassifier(nn.Module):
    def __init__(self, n_features, output_size, n_hidden, n_layers, dropout):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.linear = nn.Linear(n_hidden, output_size)

    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        logits = self.linear(last_hidden)
        return logits


class LSTMClassifierModule(BaseClassifierModule):
    """
    LSTM classifier Lightning module.

    Uses LSTM layers for sequence modeling with the final hidden state
    used for classification. Now includes accuracy tracking and confusion
    matrix visualization like other classifiers.
    """

    def __init__(
        self,
        n_features=1,
        output_size=3,
        n_hidden=64,
        n_layers=2,
        dropout=0.0,
        label_smoothing=0.0,
        lr=1e-3,
        optimizer="adam",  # Default to Adam to maintain original behavior
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

        # Create the LSTM model architecture
        self.model = LSTMClassifier(
            n_features=n_features,
            output_size=output_size,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout=dropout,
        )
