"""Session queue construction from the curves stored on ``words``."""
import math

import pytest

from src.database import Direction
from src.gui.practice.queue_model import (
    TIER_NEW,
    TIER_PRIORITY_NEW,
    TIER_REVIEW,
    build_queue,
    card_gain,
)
from src.model.config import PredictConfig
from src.model.curve import Curve, invert_curve
from tests.factories import DAY, T0, add_word

F, R = Direction.FORWARD, Direction.REVERSE
NOW = T0 + DAY
CFG = PredictConfig(recall_threshold=0.8, max_delta_seconds=math.inf)


def curves(prefix: str, tau: float, ceiling: float = 1.0) -> dict:
    """Current + post-review curve columns for one direction."""
    return {
        f"{prefix}_tau": tau, f"{prefix}_ceiling": ceiling,
        f"{prefix}_ok_tau": 10 * tau, f"{prefix}_ok_ceiling": ceiling,
        f"{prefix}_no_tau": tau / 10, f"{prefix}_no_ceiling": ceiling,
    }


def learned(source: str, tau: float, **extra) -> int:
    """A word learned in both directions at T0 with the same curve each way."""
    return add_word(
        source, source.upper(), {F: [(T0, True)], R: [(T0, True)]},
        **curves("fwd", tau), **curves("rev", tau), **extra,
    )


def drain(queue):
    out = []
    while len(queue):
        tier, priority, _, card = queue.pop_entries(1)[0]
        out.append((tier, priority, card))
    return out


def test_due_cards_go_to_main_queue_by_gain(db):
    learned("due_short", DAY / 2)
    learned("due_long", DAY)
    queue, waiting = build_queue(NOW, CFG)
    entries = drain(queue)
    assert len(waiting) == 0
    assert {tier for tier, _, _ in entries} == {TIER_REVIEW}
    scores = [c.score for _, _, c in entries]
    assert scores == sorted(scores, reverse=True)
    for _, priority, c in entries:
        assert priority == pytest.approx(-card_gain(c, NOW))


def test_not_due_cards_wait_keyed_by_due_time(db):
    learned("fresh", 100 * DAY)
    queue, waiting = build_queue(NOW, CFG)
    assert len(queue) == 0 and len(waiting) == 2
    assert waiting.peek_priority() == T0 + int(invert_curve(100 * DAY, 0.8, math.inf, 1.0))


def test_without_threshold_not_due_cards_compete_on_gain(db):
    learned("fresh", 100 * DAY)
    queue, waiting = build_queue(NOW, CFG, use_threshold=False)
    assert len(waiting) == 0 and len(queue) == 2


def test_tier_order_unscored_due_priority_new_new(db):
    learned("due", DAY / 2)
    unscored = add_word("unscored", "U", {F: [(T0, True)]})
    prio = add_word("prio", "P", priority=True)
    plain = add_word("plain", "N")
    entries = drain(build_queue(NOW, CFG)[0])

    assert entries[0][2].word_id == unscored  # unscored first, so it gets rescored
    tiers = [t for t, _, _ in entries]
    assert tiers == sorted(tiers)
    assert {c.word_id for t, _, c in entries if t == TIER_PRIORITY_NEW} == {prio}
    # the unscored word's REVERSE is new too, so it lands in the new tier
    assert {c.word_id for t, _, c in entries if t == TIER_NEW} == {plain, unscored}


def test_ignored_words_are_left_out(db):
    learned("ignored", DAY / 2, ignored=True)
    add_word("ignored_new", "X", ignored=True)
    queue, waiting = build_queue(NOW, CFG)
    assert len(queue) == 0 and len(waiting) == 0


def test_new_word_directions_are_back_to_back(db):
    ids = [add_word(f"w{i}", f"W{i}") for i in range(6)]
    order = [c.word_id for _, _, c in drain(build_queue(NOW, CFG)[0])]
    assert sorted(order) == sorted(ids * 2)
    assert all(order[i] == order[i + 1] for i in range(0, len(order), 2))


def test_half_learned_word_is_finished_before_new_pairs(db):
    for i in range(5):
        add_word(f"pair{i}", f"P{i}")
    half = add_word("half", "H", {F: [(T0, True)]}, **curves("fwd", 100 * DAY))
    entries = [e for e in drain(build_queue(NOW, CFG)[0]) if e[0] == TIER_NEW]
    assert entries[0][2].word_id == half and entries[0][2].direction is R


def test_gain_only_ordering_scores_new_cards_in_review_tier(db):
    learned("due", DAY / 2)
    prio = add_word("prio", "P", priority=True)
    plain = add_word("plain", "N")
    new_curves = {d: (Curve(1e9, 0.9), Curve(1e3, 0.5)) for d in Direction}
    entries = drain(build_queue(NOW, CFG, new_card_curves=new_curves, first_success=0.6)[0])

    assert {t for t, _, _ in entries} == {TIER_REVIEW}
    new = [c for _, _, c in entries if c.last_practiced is None]
    assert [c.word_id for c in new] == [prio, prio, plain, plain]
    # Huge post-review curves outrank the learned card's gain.
    assert entries[0][2].word_id == prio
