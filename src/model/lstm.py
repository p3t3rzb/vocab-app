"""The :class:`RecallLSTM` network definition.

A small 2-layer LSTM whose per-step output is a single sigmoid — interpreted
as ``P(remembered)`` at that timestep.
"""

import torch
import torch.nn as nn


class RecallLSTM(nn.Module):
    """Predicts ``P(remembered)`` at each step in a repetition sequence.

    Input per timestep is ``[log(Δt + 1), prev_remembered]``: the log-time
    elapsed since the previous repetition (in seconds) and whether that
    previous attempt was successful.
    """

    def __init__(
        self, hidden_size: int = 512, num_layers: int = 2, dropout: float = 0.2
    ):
        """Build the network.

        Args:
            hidden_size: LSTM hidden state size.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability — applied between LSTM layers
                (only when ``num_layers > 1``) and before the output head.
        """
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
        """Forward pass.

        Args:
            x: Padded inputs of shape ``(B, L, 2)``.

        Returns:
            Tensor of shape ``(B, L)`` with ``P(remembered)`` at each step.
            Values at padded positions are undefined; callers must mask them
            out using the per-sequence lengths.
        """
        out, _ = self.lstm(x)
        out = self.drop(out)
        logits = self.head(out).squeeze(-1)
        return torch.sigmoid(logits)

    def hyperparams(self) -> dict:
        """Return the constructor kwargs needed to rebuild this network.

        Used when saving checkpoints so :func:`load_model` can reconstruct
        the same architecture before loading the state dict.
        """
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }
