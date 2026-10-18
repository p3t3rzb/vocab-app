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

Words flagged as **priority** are introduced first. The heap is keyed by a
*tier* before the score, so a session runs: due reviews, new priority words, new
ordinary words — each by its own key within the tier. Priority only decides the
order new words are met in; once learned, a priority word's reviews compete on
expected gain like any other.

Once even those run out the session need not end: :func:`drain_waiting` folds
the not-yet-due cards into the main queue on the same expected-gain key, so
practice simply continues with whichever card the next review would help most.
The threshold, which until then decided *whether* a card is served, drops out;
the ordering does not change, because it never depended on it. The folded-in
cards share one last tier regardless of priority, so a card that comes due
again meanwhile is still served ahead of them.

Two settings change that layout (see :func:`build_queue`). **Gain-only ordering**
drops the separate new tiers: a never-practised card is scored too — its whole
expected post-review area, with the deck's first-attempt success rate standing in
for recall — and competes with the due reviews, though every new priority word
is still met before any ordinary one. **No threshold** (only offered alongside
gain-only ordering, since otherwise learned cards would never run out and no new
word would be met) drops the waiting heap: every learned card is in the main
queue on its gain.

Ordering on the gain rather than on current recall matters for a second reason:
worst-recalled-first always reviews weak cards immediately, which locks difficulty
and gap together and is why the trained model fits almost no decay. Deferring weak
cards is what supplies the long-gap observations the model needs.
"""
from __future__ import annotations

import heapq
import math
import random
from collections.abc import Callable
from dataclasses import dataclass

from src.database import (
    Direction,
    RepetitionRepository,
    WordRepository,
    get_session,
)
from src.model.config import PredictConfig
from src.model.curve import Curve, expected_gain, invert_curve, recall_at

#: Post-review ``(success, failure)`` curves of a never-practised card, per direction.
NewCardCurves = dict[Direction, tuple[Curve, Curve]]

# Priority for a card whose post-review curves have not been computed yet (a
# database predating them). Scores are negated gains in *seconds*, and a gain is
# a difference of time constants, so the deck's longest curve bounds them —
# years, i.e. ~1e8 — and no real card comes near this. Such cards are shown first, which gets
# them rescored. A random offset in [0, 1) shuffles them among themselves.
_UNSCORED_PRIORITY = -1e18

#: The opposite end of the heap, for a card the session could not score at all
#: (the answer worker hit an error). A plain ``0.0`` score no longer means "last"
#: now that a gain can be negative, so this is spelled out separately.
ERROR_PRIORITY = -_UNSCORED_PRIORITY

#: How many of the most urgent cards :func:`refresh_top` re-dates before one is
#: served. Measured on the French deck: ~30 ms per card shown at 8, ~160 ms at 32.
#: A card answered earlier in the session is the one most overstated, so it sits
#: at the head of the heap and a small window still catches it.
REFRESH_TOP_K = 8

#: Heap tiers, compared before the priority key: every card in a lower tier is
#: served before any card in a higher one. Learned cards (due or unscored) sit in
#: the review tier, never-practised ones in a new tier split by the word's
#: priority flag.
TIER_REVIEW = 0
TIER_PRIORITY_NEW = 1
TIER_NEW = 2
#: The extra-practice pool (:func:`drain_waiting`) and cards the session could not
#: score — behind everything, so a card that comes due is still served first.
TIER_EXTRA = 3


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
    #: Whether the word is in the priority subset, whose new words are met first.
    priority: bool
    #: The direction's three stored curves, each a time constant with the ceiling
    #: the model predicted beside it: the present curve (``None`` when the card has
    #: never been practised in this direction) and those of the two a remembered /
    #: forgotten answer would produce (``None`` only on a database predating them,
    #: or after the answer worker failed). Carried so :func:`card_gain` can
    #: re-score the card whenever it is served, instead of it keeping the score it
    #: was built with.
    current: Curve | None
    success: Curve | None
    failure: Curve | None


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
    """Min-heap of cards ordered by tier, then priority (lower = practiced first).

    Wraps :mod:`heapq` so both :meth:`push` and :meth:`pop` are ``O(log N)``.
    Heap entries are ``(tier, priority, seq, card)``; the monotonic ``seq``
    counter breaks ties so :class:`Card` instances are never compared directly.
    The waiting heap leaves every card at tier 0, so it is keyed by due time alone.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, float, int, Card]] = []
        self._seq = 0

    def push(self, card: Card, priority: float, tier: int = 0) -> None:
        """Insert ``card`` with the given tier and priority (lower is more urgent)."""
        heapq.heappush(self._heap, (tier, priority, self._seq, card))
        self._seq += 1

    def pop(self) -> Card | None:
        """Remove and return the lowest-priority card, or ``None`` if empty."""
        if not self._heap:
            return None
        return heapq.heappop(self._heap)[3]

    def pop_entries(self, k: int) -> list[tuple[int, float, int, Card]]:
        """Remove and return up to ``k`` raw ``(tier, priority, seq, card)`` entries, most urgent first."""
        return [heapq.heappop(self._heap) for _ in range(min(k, len(self._heap)))]

    def push_entry(self, entry: tuple[int, float, int, Card]) -> None:
        """Re-insert an entry from :meth:`pop_entries`, keeping its ``seq``.

        Keeping the original ``seq`` matters for ties: new cards share one key and
        are ordered by ``seq`` alone, so re-pushing them as fresh entries would move
        them behind every other new card and could split a word's two directions.
        """
        heapq.heappush(self._heap, entry)

    def remove_word(self, word_id: int) -> None:
        """Drop every card of ``word_id`` (either direction) from the heap.

        ``O(N)`` — a rebuild — but it only runs when the user ignores a word.
        """
        self._heap = [e for e in self._heap if e[3].word_id != word_id]
        heapq.heapify(self._heap)

    def peek_priority(self) -> float | None:
        """Return the smallest priority without popping, or ``None`` if empty."""
        if not self._heap:
            return None
        return self._heap[0][1]

    def __len__(self) -> int:
        return len(self._heap)


def _curve(word, direction: Direction, suffix: str) -> Curve | None:
    """Return one stored curve, or ``None`` if it is unset.

    ``suffix`` selects which of a direction's three curves to read: ``""`` for the
    current one, ``"_ok"`` / ``"_no"`` for the curves a remembered / forgotten
    answer right now would produce.

    The two columns are written together and are treated as one value here: if
    either is ``NULL`` the curve is missing, since half a curve cannot be
    evaluated.
    """
    prefix = "fwd" if direction is Direction.FORWARD else "rev"
    tau = getattr(word, f"{prefix}{suffix}_tau")
    p0 = getattr(word, f"{prefix}{suffix}_ceiling")
    return None if tau is None or p0 is None else Curve(tau, p0)


def card_gain(card: Card, now: int) -> float:
    """Expected retained seconds a review of ``card`` at ``now`` would add.

    The queue's ordering key, and the only place it is computed, so a card scored
    at build time, on promotion from the waiting heap, and right after an answer
    all go through the same arithmetic with the elapsed time each of those moments
    implies. A gain moves as the clock does (see
    :func:`src.model.curve.expected_gain`), so re-scoring on the way out of the
    heap matters in a way it did not when the key was a level.

    The gain integrates the curve out to infinity rather than over a finite
    retention horizon, which costs the served order nothing: the recall threshold
    already excludes the cards a horizon would affect. A horizon only bites on a
    curve whose time constant is comparable to it, and a card with one that long
    still has high recall, so it is parked in the waiting heap rather than
    served. The due pool's median time constant on the French deck is ~5 days —
    saturated to machine precision inside any horizon worth setting — and the
    order it produces is identical for the first 1201 of 1874 due cards, differing
    only in the long-abandoned tail. Where the two do diverge is the extra-practice
    phase (:func:`drain_waiting`), which serves exactly the long-τ cards a
    finite horizon was clipping.

    Args:
        card: The card to score. Must have a current curve and both post-review
            ones; callers handle a card that does not via
            :data:`_UNSCORED_PRIORITY`, and a never-practised card is never
            scored at all — it is ordered by its place among the new cards.
        now: Unix timestamp the review is hypothetically happening at.

    Returns:
        Expected added recallable seconds. Negative when the card is better left
        alone than reviewed.
    """
    if card.success is None or card.failure is None:
        raise ValueError("card has no post-review curves to score")
    if card.current is None or card.last_practiced is None:
        raise ValueError("card has never been practised, so it has no gain to score")
    recall = recall_at(
        card.current.tau, float(now - card.last_practiced), card.current.ceiling
    )
    # Only the current curve's *time constant* goes into the gain: its ceiling is
    # already inside `recall`, and applying it twice is the trap `remaining_seconds`
    # exists to prevent.
    return expected_gain(recall, card.current.tau, card.success, card.failure)


def refresh_top(
    queue: PracticeQueue,
    now: int,
    refresh: Callable[[list[Card], int], None],
    k: int = REFRESH_TOP_K,
) -> None:
    """Re-date the post-review curves of the ``k`` most urgent cards to ``now`` and re-key them.

    A card's success / failure curves come from appending a hypothetical review
    to its history, and that review's *gap* is an input to the model. The stored
    curves date it to whenever they were computed — the last full param pass, or
    the card's own last answer, i.e. a gap of zero. Scoring them against the live
    recall then mixes two moments. The model is most sensitive to that gap within
    the first hour (on the French deck the median post-lapse time constant falls
    from ~790 to ~560 days between a 0 and a 1 h gap), so a card answered earlier
    in the session comes back with an overstated gain.

    Re-dating every card would cost ~2 ms each, too slow for the whole deck, so
    only the head of the heap is refreshed just before a card is served. That is
    where it matters: a stale card overstates its gain, which is exactly what puts
    it near the head. The refreshed curves are not written back — they would go
    stale again just as quickly.

    Cards without a full set of curves (new, unscored, failed) keep their key.
    A new card's curves need no re-dating: its hypothetical review is its first,
    whose gap is always zero.

    Args:
        queue: Main queue; its head is popped, re-keyed and pushed back.
        now: Unix timestamp the hypothetical review is dated to.
        refresh: Sets ``success`` / ``failure`` on each card it is given for a
            review at ``now``.
        k: How many of the most urgent cards to refresh.
    """
    entries = queue.pop_entries(k)
    stale = [
        card
        for _, _, _, card in entries
        if card.current is not None
        and card.last_practiced is not None
        and card.success is not None
        and card.failure is not None
    ]
    if stale:
        refresh(stale, now)
    stale_ids = {id(card) for card in stale}
    for tier, priority, seq, card in entries:
        if id(card) in stale_ids:
            card.score = card_gain(card, now)
            priority = -card.score
        queue.push_entry((tier, priority, seq, card))


def drain_waiting(queue: PracticeQueue, waiting: PracticeQueue, now: int) -> int:
    """Move every waiting card into the main queue, re-scored at ``now``.

    What the session does when the user asks to keep practising after the queue
    has run dry: the cards left in the waiting heap are exactly the learned ones
    still above the recall threshold, and folding them in on the usual ``−score``
    key drops that threshold without changing the order anything is served in.

    They are re-scored rather than pushed with the score they were built with,
    for the same reason promotion re-scores them: a gain moves with the clock.

    Only the cards already waiting are folded in — answering one still parks it
    by its new due time, so each comes up once and the phase lasts as long as the
    vocabulary does. Re-queueing an answered card on its gain instead would serve
    it again almost immediately: a card reviewed a moment ago scores *high*, not
    low, because the curve a further review would leave is much longer than the
    one just fitted.

    Args:
        queue: Main queue to push into.
        waiting: Waiting queue, emptied by this call.
        now: Unix timestamp the gains are evaluated at.

    Returns:
        How many cards were moved.
    """
    moved = 0
    while True:
        card = waiting.pop()
        if card is None:
            return moved
        card.score = card_gain(card, now)
        queue.push(card, -card.score, TIER_EXTRA)
        moved += 1


def park_not_due(
    queue: PracticeQueue,
    waiting: PracticeQueue,
    card: Card,
    due_ts: int,
    use_threshold: bool,
) -> None:
    """Place a scored learned card that is above the recall threshold.

    With the threshold on it waits in ``waiting`` until ``due_ts``. Without it, it
    competes in the main queue on its gain (``card.score``) in the review tier.
    """
    if use_threshold:
        waiting.push(card, due_ts)
    else:
        queue.push(card, -card.score, TIER_REVIEW)


def build_queue(
    now: int,
    cfg: PredictConfig,
    *,
    use_threshold: bool = True,
    new_card_curves: NewCardCurves | None = None,
    first_success: float = 0.5,
) -> tuple[PracticeQueue, PracticeQueue]:
    """Build the (main, waiting) practice queues.

    Both queues are filled from stored time constants only — no model is loaded and no
    forward is run, so this stays fast enough to sit in front of the first card.
    The main queue holds the learned cards that are due, ordered by ``−score``
    (highest expected gain practiced first), with the new cards behind them; the
    waiting queue holds not-due cards keyed by their due timestamp (soonest first)
    so the session can promote them as they come due.

    For each (word, direction):

    * **New** (never practiced in that direction) → main queue in a new tier,
      i.e. behind every due learned card, so new words are only met once
      everything learned is above the threshold. New cards are
      grouped per word and shuffled as pairs, so both directions of a freshly-seen
      word appear back to back (in random order within the pair). Words whose
      *other* direction was already learned (only one new direction left) come
      first, so a half-learned word is finished before brand-new pairs are
      introduced.
    * **Due** (``recall ≤ threshold``) → main queue at ``−score`` in the review
      tier. The threshold
      only gates *whether* a card is served, never the order the served ones come
      in.
    * **Not due** (``recall > threshold``) → waiting queue keyed by its due
      timestamp ``last + invert_curve(...)``, always ``> now``.
    * **Unscored** (curves not computed yet, i.e. history recorded with no
      estimator) → main queue at :data:`_UNSCORED_PRIORITY`, i.e. first within
      the review tier, so they get rescored on answer.

    Priority only orders the new cards (see the ``TIER_*`` constants): due
    reviews, then new priority words, then new ordinary words. A learned card is
    handled the same whatever its word's priority.

    Ignored words are left out of both queues.

    Every learned card carries the curves it was scored from, so the session can
    re-score it with :func:`card_gain` at the moment it is actually served.

    **Gain-only ordering** (``new_card_curves`` given): new cards skip the new
    tiers and go into the review tier at ``−score`` like a due card, scored by
    :func:`~src.model.curve.expected_gain` with no baseline. A word's new cards
    are still kept back to back: both are keyed by the *mean* of their gains (the
    pair costs two reviews), and equal keys pop in push order. A half-learned
    word's remaining card is keyed by its own gain. All new cards of a direction
    share the same curves, so words are shuffled to make ties fall in random
    order. Priority words are pushed first, and an ordinary word's key is raised
    to at least the last priority word's — new-card keys never change, so every
    new priority word is met before any ordinary one.

    **No threshold** (``use_threshold=False``): not-due cards go into the main
    queue via :func:`park_not_due` instead of waiting, so the waiting heap stays
    empty.

    Args:
        now: Unix timestamp the recall scores are evaluated at.
        cfg: Supplies the recall threshold and the ``max_delta_seconds`` cap on
            a waiting card's due time.
        use_threshold: Whether not-due cards wait (``True``) or compete on gain.
        new_card_curves: Post-review curves of a never-practised card per
            direction, which switches on gain-only ordering; ``None`` keeps new
            cards trailing the learned ones.
        first_success: The deck's first-attempt success rate, standing in for a
            new card's recall. Only read with ``new_card_curves``.
    """
    queue = PracticeQueue()
    waiting = PracticeQueue()
    new_by_word: dict[int, list[Card]] = {}
    new_priority_by_word: dict[int, list[Card]] = {}
    new_scored_by_word: dict[int, list[Card]] = {}
    new_scored_priority_by_word: dict[int, list[Card]] = {}
    with get_session() as session:
        words = WordRepository(session).get_all()
        last_by_dir = RepetitionRepository(session).latest_practiced_at_by_word_direction()

    for word in words:
        if word.ignored:
            continue
        for direction in Direction:
            last = last_by_dir.get((word.id, int(direction)))
            current = _curve(word, direction, "")

            card = Card(
                word_id=word.id,
                direction=direction,
                source_text=word.source_text,
                target_text=word.target_text,
                last_practiced=last,
                score=0.0,
                priority=word.priority,
                current=current,
                success=_curve(word, direction, "_ok"),
                failure=_curve(word, direction, "_no"),
            )

            if last is None and new_card_curves is not None:
                # Gain-only ordering: scored like any other card. Collected per
                # word so a word's two directions can be pushed back to back.
                card.success, card.failure = new_card_curves[direction]
                card.score = expected_gain(
                    first_success, None, card.success, card.failure
                )
                new = new_scored_priority_by_word if word.priority else new_scored_by_word
                new.setdefault(word.id, []).append(card)
                continue

            if last is None:
                # Never practiced in this direction — trails all learned cards.
                # Collect per word so both directions can be kept together.
                new = new_priority_by_word if word.priority else new_by_word
                new.setdefault(word.id, []).append(card)
                continue

            if current is None or card.success is None or card.failure is None:
                # Practiced, but without the curves to score it from.
                queue.push(card, _UNSCORED_PRIORITY + random.random(), TIER_REVIEW)
                continue

            card.score = card_gain(card, now)

            if recall_at(current.tau, now - last, current.ceiling) <= cfg.recall_threshold:
                queue.push(card, -card.score, TIER_REVIEW)
            else:
                # Not due yet — park it in the waiting heap keyed by due time.
                due_ts = last + int(
                    invert_curve(
                        current.tau,
                        cfg.recall_threshold,
                        cfg.max_delta_seconds,
                        current.ceiling,
                    )
                )
                park_not_due(queue, waiting, card, due_ts, use_threshold)

    floor = -math.inf  # ordinary new words key no earlier than any priority one
    for new_scored in (new_scored_priority_by_word, new_scored_by_word):
        new_words = list(new_scored.values())
        random.shuffle(new_words)
        keys = [max(-sum(c.score for c in cards) / len(cards), floor) for cards in new_words]
        for cards, key in zip(new_words, keys):
            random.shuffle(cards)  # random direction order within the pair
            for card in cards:
                queue.push(card, key, TIER_REVIEW)
        floor = max(keys, default=floor)
    _push_new(queue, new_priority_by_word, TIER_PRIORITY_NEW)
    _push_new(queue, new_by_word, TIER_NEW)

    return queue, waiting


def _push_new(queue: PracticeQueue, new_by_word: dict[int, list[Card]], tier: int) -> None:
    """Push one priority group's new cards into ``tier``, word by word.

    The tier puts them after every due learned card. A single-card
    bucket means the word's other direction already has history (learned in an
    earlier session), so finish that half-learned word before meeting brand-new
    pairs: order the single-card buckets ahead of the two-card ones. Shuffle
    within each group (randomizes order) and push each bucket's cards
    contiguously with an increasing key so both directions of a word land back
    to back.
    """
    buckets = list(new_by_word.values())
    partial = [b for b in buckets if len(b) == 1]  # other direction already learned
    pairs = [b for b in buckets if len(b) != 1]     # both directions still new
    random.shuffle(partial)
    random.shuffle(pairs)
    for i, cards in enumerate(partial + pairs):
        random.shuffle(cards)  # random direction order within the pair
        for j, card in enumerate(cards):
            queue.push(card, i + j * 0.5, tier)
