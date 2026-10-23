"""Practice-queue primitives that don't need a database: the heap, scoring, refresh and drain."""
import pytest

from src.database import Direction
from src.gui.practice.queue_model import (
    TIER_EXTRA,
    TIER_REVIEW,
    Card,
    PracticeQueue,
    card_gain,
    drain_waiting,
    park_not_due,
    refresh_top,
)
from src.gui.practice.state import ArrowKey
from src.model.curve import Curve, expected_gain, recall_at
from tests.factories import DAY, T0


def card(word_id: int = 1, direction=Direction.FORWARD, **overrides) -> Card:
    fields = dict(
        word_id=word_id,
        direction=direction,
        source_text="chat",
        target_text="kot",
        last_practiced=T0,
        score=0.0,
        priority=False,
        current=Curve(5 * DAY, 0.9),
        success=Curve(20 * DAY, 0.9),
        failure=Curve(DAY, 0.5),
    )
    fields.update(overrides)
    return Card(**fields)


class TestCard:
    def test_forward_prompts_source(self):
        c = card()
        assert (c.prompt_text(), c.answer_text()) == ("chat", "kot")
        assert c.direction_label("French", "Polish") == "French → Polish"

    def test_reverse_prompts_target(self):
        c = card(direction=Direction.REVERSE)
        assert (c.prompt_text(), c.answer_text()) == ("kot", "chat")
        assert c.direction_label("French", "Polish") == "Polish → French"


class TestPracticeQueue:
    def test_pops_by_tier_then_priority(self):
        q = PracticeQueue()
        q.push(card(1), priority=-5.0, tier=2)
        q.push(card(2), priority=10.0, tier=0)
        q.push(card(3), priority=-1.0, tier=0)
        assert [q.pop().word_id for _ in range(3)] == [3, 2, 1]
        assert q.pop() is None

    def test_ties_pop_in_push_order(self):
        q = PracticeQueue()
        for i in range(5):
            q.push(card(i), priority=0.0)
        assert [q.pop().word_id for _ in range(5)] == list(range(5))

    def test_pop_entries_and_push_entry_preserve_seq(self):
        q = PracticeQueue()
        for i in range(4):
            q.push(card(i), priority=0.0)
        entries = q.pop_entries(2)
        assert len(q) == 2
        for e in entries:
            q.push_entry(e)
        assert [q.pop().word_id for _ in range(4)] == [0, 1, 2, 3]

    def test_remove_word_drops_both_directions(self):
        q = PracticeQueue()
        q.push(card(1, Direction.FORWARD), 0.0)
        q.push(card(1, Direction.REVERSE), 1.0)
        q.push(card(2), 2.0)
        q.remove_word(1)
        assert len(q) == 1 and q.pop().word_id == 2

    def test_peek_priority(self):
        q = PracticeQueue()
        assert q.peek_priority() is None
        q.push(card(), 3.0)
        q.push(card(), 1.0)
        assert q.peek_priority() == 1.0


class TestCardGain:
    def test_matches_expected_gain_with_live_recall(self):
        c = card()
        now = T0 + 3 * DAY
        recall = recall_at(5 * DAY, 3 * DAY, 0.9)
        assert card_gain(c, now) == pytest.approx(
            expected_gain(recall, 5 * DAY, c.success, c.failure)
        )

    def test_requires_post_review_curves(self):
        with pytest.raises(ValueError):
            card_gain(card(success=None), T0)

    def test_requires_history(self):
        with pytest.raises(ValueError):
            card_gain(card(current=None, last_practiced=None), T0)


def test_park_not_due_with_threshold_waits_by_due_time():
    q, waiting = PracticeQueue(), PracticeQueue()
    park_not_due(q, waiting, card(), due_ts=T0 + DAY, use_threshold=True)
    assert len(q) == 0 and waiting.peek_priority() == T0 + DAY


def test_park_not_due_without_threshold_competes_on_gain():
    q, waiting = PracticeQueue(), PracticeQueue()
    c = card(score=123.0)
    park_not_due(q, waiting, c, due_ts=T0 + DAY, use_threshold=False)
    assert len(waiting) == 0 and q.peek_priority() == -123.0


def test_drain_waiting_rescores_into_extra_tier():
    q, waiting = PracticeQueue(), PracticeQueue()
    q.push(card(99), priority=1e9, tier=TIER_REVIEW)  # a due card stays ahead
    for i in range(3):
        waiting.push(card(i, last_practiced=T0 - i * DAY), priority=T0 + i)
    now = T0 + DAY
    assert drain_waiting(q, waiting, now) == 3
    assert len(waiting) == 0
    assert q.pop().word_id == 99
    rest = [q.pop() for _ in range(3)]
    assert [c.score for c in rest] == sorted((card_gain(c, now) for c in rest), reverse=True)
    entries = [(TIER_EXTRA, -c.score) for c in rest]
    assert entries == sorted(entries)


def test_refresh_top_rescores_only_fully_curved_cards():
    q = PracticeQueue()
    scored = card(1, score=1.0)
    new = card(2, current=None, last_practiced=None, success=None, failure=None)
    q.push(scored, -1.0)
    q.push(new, 5.0)
    calls = []

    def refresh(cards, now):
        calls.append([c.word_id for c in cards])
        for c in cards:
            c.success = Curve(40 * DAY, 0.95)

    now = T0 + 2 * DAY
    refresh_top(q, now, refresh, k=8)
    assert calls == [[1]]
    assert scored.score == pytest.approx(card_gain(scored, now))
    assert len(q) == 2


def test_refresh_top_limits_to_k():
    q = PracticeQueue()
    for i in range(5):
        q.push(card(i), float(i))
    seen = []
    refresh_top(q, T0 + DAY, lambda cards, now: seen.extend(c.word_id for c in cards), k=2)
    assert seen == [0, 1]
    assert len(q) == 5


def test_arrow_key_from_str():
    assert ArrowKey.from_str("Left") is ArrowKey.LEFT
    assert ArrowKey.from_str("Escape") is None
