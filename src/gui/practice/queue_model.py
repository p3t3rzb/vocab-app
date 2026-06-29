"""Practice queue data model.

A :class:`Card` is one ``(word, direction)`` pair the user will see during a
session. The queue is a :class:`PracticeQueue` — a min-heap keyed by *live recall
score* (lower = more likely forgotten = shown first), so due words are practiced
worst-first. Never-practiced words trail all due ones. A card the predictor still
marks due-now after a failed attempt is pushed back in at its correct position.
"""
from __future__ import annotations

import heapq
import random
from dataclasses import dataclass

from src.database import (
    Direction,
    RepetitionRepository,
    WordRepository,
    get_session,
)
from src.model.config import PredictConfig
from src.model.curve import invert_curve, recall_at

# Priority floor for never-practiced ("new") cards: always after any due card,
# whose recall score lives in (0, 1). The random offset shuffles new cards.
_NEW_PRIORITY_BASE = 1.0


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


class PracticeQueue:
    """Min-heap of cards ordered by priority (lower = practiced first).

    Wraps :mod:`heapq` so both :meth:`push` and :meth:`pop` are ``O(log N)``.
    Heap entries are ``(priority, seq, card)``; the monotonic ``seq`` counter
    breaks ties so :class:`Card` instances are never compared directly.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, Card]] = []
        self._seq = 0

    def push(self, card: Card, priority: float) -> None:
        """Insert ``card`` with the given priority (lower is more urgent)."""
        heapq.heappush(self._heap, (priority, self._seq, card))
        self._seq += 1

    def pop(self) -> Card | None:
        """Remove and return the lowest-priority card, or ``None`` if empty."""
        if not self._heap:
            return None
        return heapq.heappop(self._heap)[2]

    def peek_priority(self) -> float | None:
        """Return the smallest priority without popping, or ``None`` if empty."""
        if not self._heap:
            return None
        return self._heap[0][0]

    def __len__(self) -> int:
        return len(self._heap)


def _direction_params(word, direction: Direction) -> tuple[float, float, float] | None:
    """Return the stored ``(p0, S, d)`` curve params for a direction, or ``None``."""
    if direction is Direction.FORWARD:
        trio = (word.fwd_p0, word.fwd_s, word.fwd_d)
    else:
        trio = (word.rev_p0, word.rev_s, word.rev_d)
    if any(v is None for v in trio):
        return None
    return trio  # type: ignore[return-value]


def build_queue(now: int, cfg: PredictConfig) -> tuple[PracticeQueue, PracticeQueue]:
    """Build the (main, waiting) practice queues.

    The main queue is ordered by live recall score (lower = more likely
    forgotten = shown first); the waiting queue holds not-due cards keyed by
    their due timestamp (soonest first) so the session can promote them as they
    come due. For each (word, direction):

    * **New** (never practiced in that direction) → main queue, after every due
      card, in random order.
    * **Due, scored** (has history + stored params, ``recall ≤ threshold``) →
      main queue with ``priority = recall`` so the worst-recalled comes first.
    * **Due, unscored** (history but no params, i.e. no trained model) →
      main queue with a random priority in ``[0, 1)`` (can't score it).
    * **Not due** (scored, ``recall > threshold``) → waiting queue keyed by its
      due timestamp ``last + invert_curve(...)`` (always ``> now``).
    """
    queue = PracticeQueue()
    waiting = PracticeQueue()
    with get_session() as session:
        words = WordRepository(session).get_all()
        last_by_dir = RepetitionRepository(session).latest_practiced_at_by_word_direction()

    for word in words:
        for direction in Direction:
            last = last_by_dir.get((word.id, int(direction)))
            card = Card(
                word_id=word.id,
                direction=direction,
                source_text=word.source_text,
                target_text=word.target_text,
                last_practiced=last,
            )
            if last is None:
                # Never practiced in this direction — trails all due cards.
                queue.push(card, _NEW_PRIORITY_BASE + random.random())
                continue

            params = _direction_params(word, direction)
            if params is None:
                # Has history but no model-computed params — treat as due.
                queue.push(card, random.random())
                continue

            p0, s, d = params
            recall = recall_at(p0, s, d, now - last)
            if recall <= cfg.recall_threshold:
                queue.push(card, recall)
            else:
                # Not due yet — park it in the waiting heap keyed by due time.
                due_ts = last + int(
                    invert_curve(p0, s, d, cfg.recall_threshold, cfg.max_delta_seconds)
                )
                waiting.push(card, due_ts)

    return queue, waiting
