"""Builders for test data: transient repetitions and rows inserted into the test DB."""

from __future__ import annotations

from collections.abc import Iterable

from src.database import Direction, Repetition, Word, get_session

DAY = 86_400
T0 = 1_700_000_000  # an arbitrary fixed "epoch" so tests never depend on the clock


def make_reps(
    events: Iterable[tuple[int, bool]],
    word_id: int = 1,
    direction: Direction = Direction.FORWARD,
) -> list[Repetition]:
    """Transient (unsaved) repetitions from ``(practiced_at, remembered)`` pairs."""
    return [
        Repetition(
            word_id=word_id,
            direction=int(direction),
            practiced_at=t,
            remembered=remembered,
        )
        for t, remembered in events
    ]


def add_word(
    source: str,
    target: str,
    history: dict[Direction, list[tuple[int, bool]]] | None = None,
    **columns,
) -> int:
    """Insert a word (plus optional per-direction history) and return its id.

    ``columns`` sets any other :class:`Word` attribute, e.g. ``fwd_tau=...``.
    """
    with get_session() as session:
        word = Word(source_text=source, target_text=target, **columns)
        session.add(word)
        session.flush()
        for direction, events in (history or {}).items():
            session.add_all(make_reps(events, word.id, direction))
        return word.id
