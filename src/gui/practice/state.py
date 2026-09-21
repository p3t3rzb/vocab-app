"""State machine for the practice screen.

Each card walks through the prompt → answer → saving → result transitions.
``LOADING`` is the initial state until the queue is built. Once the queue
empties, ``OFFER_EXTRA`` asks whether to keep practising cards that are not
due yet, and ``DONE`` is the terminal state when there is nothing left to
offer. ``PracticeState`` holds the states; ``ArrowKey`` is the set of arrow
keys the screen interprets.
"""
from __future__ import annotations

from enum import Enum


class PracticeState(Enum):
    """States of the practice screen's per-card flow."""

    LOADING = "loading"
    PROMPT = "prompt"
    ANSWER = "answer"
    SAVING = "saving"
    RESULT = "result"
    OFFER_EXTRA = "offer_extra"
    DONE = "done"


class ArrowKey(Enum):
    """The four arrow keys the practice screen interprets."""

    DOWN = "Down"
    UP = "Up"
    LEFT = "Left"
    RIGHT = "Right"

    @classmethod
    def from_str(cls, name: str) -> "ArrowKey | None":
        """Return the matching enum value, or ``None`` for unknown keys."""
        try:
            return cls(name)
        except ValueError:
            return None
