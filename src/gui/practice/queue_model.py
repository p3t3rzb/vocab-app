"""Practice queue data model.

A :class:`Card` is one ``(word, direction)`` pair the user will see during a
session. The queue is a :class:`PracticeQueue` — a min-heap keyed by ``−score``,
where ``score`` is the card's *expected gain*: how much future recall one review
right now would **add** over leaving the card alone
(:func:`src.model.curve.expected_gain`). Practising the highest-gaining card first
is what greedily maximises the total recall held across the vocabulary, since a
card that holds its recall unpractised contributes that recall either way — only
the part a review adds is worth spending the session on.

That ordering covers the cards already learned. Never-practised ones do not
compete in it: they trail every learned card, so a new word is only met once
everything already learned sits above the recall threshold. They are introduced
in directional pairs — both directions of a word back to back, and words with
only one direction left new ahead of brand-new pairs — so a word is finished
rather than half-learned.

Ordering on the gain rather than on current recall matters for a second reason:
worst-recalled-first always reviews weak cards immediately, which locks difficulty
and gap together and is why the trained model fits almost no decay. Deferring weak
cards is what supplies the long-gap observations the model needs.
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
from src.model.curve import expected_gain, invert_curve, recall_at

# Priority for a card whose post-review curves have not been computed yet (a
# database predating them). Scores are negated gains in *seconds*, bounded by the
# horizon, so no real card comes near this; such cards are shown first, which gets
# them rescored. A random offset in [0, 1) shuffles them among themselves.
_UNSCORED_PRIORITY = -1e18

#: Priority floor for never-practised ("new") cards: always after any learned
#: card, whose priority is a negated gain in seconds and so bounded by the
#: retention horizon. Each new card sits at this base plus its own offset.
_NEW_PRIORITY_BASE = 1e15

#: The opposite end of the heap, for a card the session could not score at all
#: (the answer worker hit an error). A plain ``0.0`` score no longer means "last"
#: now that a gain can be negative, so this is spelled out separately.
ERROR_PRIORITY = -_UNSCORED_PRIORITY


@dataclass(slots=True)
class Card:
    """One ``(word, direction)`` cell in the practice queue."""

    word_id: int
    direction: Direction
    source_text: str
    target_text: str
    last_practiced: int | None
    #: Expected retained seconds a review right now would add. The heap is keyed
    #: by its negation, so higher is practiced sooner.
    score: float
    #: The direction's three stored half-lives: the present curve's (``None`` when
    #: the card has never been practised in this direction) and those of the two a
    #: remembered / forgotten answer would produce (``None`` only on a database
    #: predating them, or after the answer worker failed). Carried so
    #: :func:`card_gain` can re-score the card whenever it is served, instead of it
    #: keeping the score it was built with.
    current: float | None
    success: float | None
    failure: float | None

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


def _half_life(word, direction: Direction, suffix: str) -> float | None:
    """Return one stored half-life, or ``None`` if it is unset.

    ``suffix`` selects which of a direction's three curves to read: ``""`` for the
    current one, ``"_ok"`` / ``"_no"`` for the curves a remembered / forgotten
    answer right now would produce.
    """
    prefix = "fwd" if direction is Direction.FORWARD else "rev"
    return getattr(word, f"{prefix}{suffix}_h")


def card_gain(card: Card, now: int, horizon_seconds: float) -> float:
    """Expected retained seconds a review of ``card`` at ``now`` would add.

    The queue's ordering key, and the only place it is computed, so a card scored
    at build time, on promotion from the waiting heap, and right after an answer
    all go through the same arithmetic with the elapsed time each of those moments
    implies. A gain moves as the clock does (see
    :func:`src.model.curve.expected_gain`), so re-scoring on the way out of the
    heap matters in a way it did not when the key was a level.

    Args:
        card: The card to score. Must have a current curve and both post-review
            ones; callers handle a card that does not via
            :data:`_UNSCORED_PRIORITY`, and a never-practised card is never
            scored at all — it is ordered by its place among the new cards.
        now: Unix timestamp the review is hypothetically happening at.
        horizon_seconds: Window the gain is integrated over.

    Returns:
        Expected added recallable seconds. Negative when the card is better left
        alone than reviewed.
    """
    if card.success is None or card.failure is None:
        raise ValueError("card has no post-review curves to score")
    if card.current is None or card.last_practiced is None:
        raise ValueError("card has never been practised, so it has no gain to score")
    elapsed = float(now - card.last_practiced)
    recall = recall_at(card.current, elapsed)
    return expected_gain(
        recall, card.current, elapsed, card.success, card.failure, horizon_seconds
    )


def build_queue(now: int, cfg: PredictConfig) -> tuple[PracticeQueue, PracticeQueue]:
    """Build the (main, waiting) practice queues.

    Both queues are filled from stored half-lives only — no model is loaded and no
    forward is run, so this stays fast enough to sit in front of the first card.
    The main queue holds the learned cards that are due, ordered by ``−score``
    (highest expected gain practiced first), with the new cards behind them; the
    waiting queue holds not-due cards keyed by their due timestamp (soonest first)
    so the session can promote them as they come due.

    For each (word, direction):

    * **New** (never practiced in that direction) → main queue at
      :data:`_NEW_PRIORITY_BASE`, i.e. behind every learned card, so new words are
      only met once everything learned is above the threshold. New cards are
      grouped per word and shuffled as pairs, so both directions of a freshly-seen
      word appear back to back (in random order within the pair). Words whose
      *other* direction was already learned (only one new direction left) come
      first, so a half-learned word is finished before brand-new pairs are
      introduced.
    * **Due** (``recall ≤ threshold``) → main queue at ``−score``. The threshold
      only gates *whether* a card is served, never the order the served ones come
      in.
    * **Not due** (``recall > threshold``) → waiting queue keyed by its due
      timestamp ``last + invert_curve(...)``, always ``> now``.
    * **Unscored** (curves not computed yet, i.e. history recorded with no
      estimator) → main queue at :data:`_UNSCORED_PRIORITY`, i.e. first, so they
      get rescored on answer.

    Every learned card carries the curves it was scored from, so the session can
    re-score it with :func:`card_gain` at the moment it is actually served.

    Args:
        now: Unix timestamp the recall scores are evaluated at.
        cfg: Supplies the recall threshold and the ``max_delta_seconds`` that
            doubles as the retention horizon.
    """
    queue = PracticeQueue()
    waiting = PracticeQueue()
    new_by_word: dict[int, list[Card]] = {}
    with get_session() as session:
        words = WordRepository(session).get_all()
        last_by_dir = RepetitionRepository(session).latest_practiced_at_by_word_direction()

    horizon = cfg.max_delta_seconds

    for word in words:
        for direction in Direction:
            last = last_by_dir.get((word.id, int(direction)))
            current = _half_life(word, direction, "")

            card = Card(
                word_id=word.id,
                direction=direction,
                source_text=word.source_text,
                target_text=word.target_text,
                last_practiced=last,
                score=0.0,
                current=current,
                success=_half_life(word, direction, "_ok"),
                failure=_half_life(word, direction, "_no"),
            )

            if last is None:
                # Never practiced in this direction — trails all learned cards.
                # Collect per word so both directions can be kept together.
                new_by_word.setdefault(word.id, []).append(card)
                continue

            if current is None or card.success is None or card.failure is None:
                # Practiced, but without the curves to score it from.
                queue.push(card, _UNSCORED_PRIORITY + random.random())
                continue

            card.score = card_gain(card, now, horizon)

            if recall_at(current, now - last) <= cfg.recall_threshold:
                queue.push(card, -card.score)
            else:
                # Not due yet — park it in the waiting heap keyed by due time.
                due_ts = last + int(
                    invert_curve(current, cfg.recall_threshold, cfg.max_delta_seconds)
                )
                waiting.push(card, due_ts)

    # Emit new cards last, after every learned card. A single-card bucket means
    # the word's other direction already has history (learned in an earlier
    # session), so finish that half-learned word before meeting brand-new pairs:
    # order the single-card buckets ahead of the two-card ones. Shuffle within
    # each group (randomizes order) and push each bucket's cards contiguously with
    # an increasing base so both directions of a word land back to back.
    buckets = list(new_by_word.values())
    partial = [b for b in buckets if len(b) == 1]  # other direction already learned
    pairs = [b for b in buckets if len(b) != 1]     # both directions still new
    random.shuffle(partial)
    random.shuffle(pairs)
    for i, cards in enumerate(partial + pairs):
        random.shuffle(cards)  # random direction order within the pair
        for j, card in enumerate(cards):
            queue.push(card, _NEW_PRIORITY_BASE + i + j * 0.5)

    return queue, waiting
