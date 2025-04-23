from torch import nn

class GRUModel(nn.Module):
    """GRU model for forex prediction."""

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0, bidirectional=False):
        """
        Initialize the GRU model.

        Args:
            input_size (int): Size of input features.
            hidden_size (int): Size of hidden layers.
            num_layers (int): Number of GRU layers.
            output_size (int): Size of output.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            bidirectional (bool, optional): Whether to use bidirectional GRU. Defaults to False.
        """
        super().__init__()

        self.model_type = "gru"
        self.meta_data = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "output_size": output_size,
            "dropout": dropout,
            "bidirectional": bidirectional
        }

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x):
        """
        Forward pass of the GRU model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # GRU forward pass
        # x shape: (batch_size, seq_len, input_size)
        gru_out, _ = self.gru(x)
        # gru_out shape: (batch_size, seq_len, hidden_size * num_directions)

        # Use the last time step output
        gru_out = gru_out[:, -1, :]
        # gru_out shape: (batch_size, hidden_size * num_directions)

        # Apply dropout
        gru_out = self.dropout(gru_out)

        # Fully connected layer
        out = self.fc(gru_out)
        # out shape: (batch_size, output_size)

        return out
