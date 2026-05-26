"""Practice queue data model.

A :class:`Card` is one ``(word, direction)`` pair the user will see during
a session. The queue is built once on entry: review-due cards first
(shuffled), then never-practiced cards (shuffled). Cards that the
predictor still marks due-now after a failed attempt are re-appended
live.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from src.database import (
    Direction,
    RepetitionRepository,
    WordRepository,
    get_session,
)


@dataclass(slots=True)
class Card:
    """One ``(word, direction)`` cell in the practice queue."""

    word_id: int
    direction: Direction
    source_text: str
    target_text: str
    last_practiced: int | None

    def prompt_text(self) -> str:
        """Text shown before the answer is revealed."""
        return self.source_text if self.direction is Direction.FORWARD else self.target_text

    def answer_text(self) -> str:
        """Text shown after the user requests the reveal."""
        return self.target_text if self.direction is Direction.FORWARD else self.source_text

    def direction_label(self, src_lang: str, tgt_lang: str) -> str:
        """``"French → Polish"`` / ``"Polish → French"`` style direction label."""
        if self.direction is Direction.FORWARD:
            return f"{src_lang} → {tgt_lang}"
        return f"{tgt_lang} → {src_lang}"


def build_queue(now: int) -> list[Card]:
    """Build the full ``review + new`` practice queue.

    Each list is shuffled independently; the concatenation order ensures
    review cards are always shown before any never-practiced new word.
    """
    review: list[Card] = []
    new: list[Card] = []
    with get_session() as session:
        words = WordRepository(session).get_all()
        reps_repo = RepetitionRepository(session)
        for word in words:
            for direction, due_ts in (
                (Direction.FORWARD, word.next_rep_fwd_at),
                (Direction.REVERSE, word.next_rep_rev_at),
            ):
                latest = reps_repo.get_latest_for_word(word.id, direction)
                card = Card(
                    word_id=word.id,
                    direction=direction,
                    source_text=word.source_text,
                    target_text=word.target_text,
                    last_practiced=latest.practiced_at if latest else None,
                )
                if latest is None:
                    new.append(card)
                elif due_ts is None or due_ts <= now:
                    review.append(card)

    random.shuffle(review)
    random.shuffle(new)
    return review + new
