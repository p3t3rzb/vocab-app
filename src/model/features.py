"""Construction of the LSTM's input rows, shared by every inference path.

:class:`~src.model.lstm.RecallLSTM` consumes one row per repetition::

    [log(gap-before-this-rep + 1), remembered, not_remembered, is_forward, is_reverse]

The model is fed the **history only** — never the gap it is being asked about —
so its output stays a clean, invertible function of ``Δt`` (see
:mod:`src.model.curve`). :func:`history_rows` builds exactly that history block;
:func:`rep_row` builds one row on its own, which is what lets callers append a
*hypothetical* next repetition and ask what the curve would look like after it.
"""
from __future__ import annotations

import math

from src.database import Direction
from src.database.models import Repetition


def rep_row(gap_seconds: float, remembered: bool, direction: Direction) -> list[float]:
    """One input row for a rep with ``remembered`` outcome, ``gap_seconds`` after the previous one.

    The direction one-hot is repeated on every row (not just carried in the
    initial hidden state) so the LSTM can condition its gates on direction over
    arbitrarily long histories.
    """
    is_rev = float(int(direction))
    rem = float(remembered)
    return [math.log(max(gap_seconds, 0.0) + 1.0), rem, 1.0 - rem, 1.0 - is_rev, is_rev]


def history_rows(reps: list[Repetition], direction: Direction) -> list[list[float]]:
    """History-only input block: one :func:`rep_row` per rep, oldest first.

    The first row has no preceding rep, so its gap is ``0``.
    """
    return [
        rep_row(
            0.0 if i == 0 else rep.practiced_at - reps[i - 1].practiced_at,
            rep.remembered,
            direction,
        )
        for i, rep in enumerate(reps)
    ]
