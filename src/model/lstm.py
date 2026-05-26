import torch
import torch.nn as nn


class RecallLSTM(nn.Module):
    """
    Predicts P(remembered) at each step in a repetition sequence.
    Input per timestep: [log(Δt + 1), prev_remembered]
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, L, 2)  padded input sequences
        returns: (B, L)     P(remembered) at each step, padded positions undefined
        """
        out, _ = self.lstm(x)
        out = self.drop(out)
        logits = self.head(out).squeeze(-1)
        return torch.sigmoid(logits)

    def hyperparams(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }
