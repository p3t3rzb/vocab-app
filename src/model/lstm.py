"""The :class:`RecallLSTM` network definition.

A small 2-layer LSTM that predicts, at each step in a repetition sequence, the
*parameters of a forgetting curve* ``R(Δt) = p0·(1 + Δt/S)**(−d)`` (see
:mod:`src.model.curve`) rather than ``P(remembered)`` directly. Recall
probability is obtained by evaluating that curve at the queried gap, and the
next-review time is found by inverting the curve analytically.

Because the curve must be a clean, invertible function of the gap, the network
is fed the repetition *history only* — never the gap being queried — so its
three raw outputs depend on past events alone.
"""

import torch
import torch.nn as nn


class RecallLSTM(nn.Module):
    """Predicts forgetting-curve parameters at each step in a repetition sequence.

    Input per timestep is ``[log(Δt_prev + 1), prev_remembered,
    prev_not_remembered, is_forward, is_reverse]``: the log-time elapsed before
    the *previous* repetition, a one-hot encoding of whether that attempt was
    successful (the history-only, gap-shifted input), and a one-hot encoding of
    the practice direction. Direction is constant across the sequence but
    supplied at every step so the LSTM can condition its dynamics on it without
    relying on the initial state surviving long histories. The output is three
    raw channels per step, turned into ``(p0, S, d)`` by :mod:`src.model.curve`.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        input_size: int = 5,
    ):
        """Build the network.

        Args:
            hidden_size: LSTM hidden state size.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability — applied between LSTM layers
                (only when ``num_layers > 1``) and before the output head.
            input_size: Number of input features per timestep. Recorded in the
                checkpoint so older models with a different feature count are
                rebuilt (and warm-start mismatches detected) correctly.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_size = input_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Padded history inputs of shape ``(B, L, input_size)``.

        Returns:
            Raw curve parameters of shape ``(B, L, 3)``. The activations and the
            curve evaluation live in :func:`src.model.curve.curve_recall`;
            values at padded positions are undefined and must be masked by the
            caller using the per-sequence lengths.
        """
        out, _ = self.lstm(x)
        out = self.drop(out)
        return self.head(out)

    def hyperparams(self) -> dict:
        """Return the constructor kwargs needed to rebuild this network.

        Used when saving checkpoints so :func:`load_model` can reconstruct
        the same architecture before loading the state dict.
        """
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "input_size": self.input_size,
        }
