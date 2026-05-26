"""Direction enum — which way a (word, repetition) pair is being practiced."""
import enum


class Direction(enum.IntEnum):
    """Which way a word is being practiced.

    Stored as an integer in the database so that SQL queries can compare
    directly against :class:`Repetition.direction` without an enum cast.
    """

    FORWARD = 0  # source → target
    REVERSE = 1  # target → source
