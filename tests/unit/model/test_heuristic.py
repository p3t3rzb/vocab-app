"""SM-2-style fallback estimator: interval ladder expressed as a curve."""
import math

import pytest

from src.database import Direction
from src.model.config import HeuristicConfig, PredictConfig
from src.model.curve import NO_CEILING, invert_curve
from src.model.inference import HeuristicPredictor
from tests.factories import DAY, T0, make_reps

H = HeuristicConfig()


def interval_of(reps) -> float:
    """The interval the heuristic targets, read back off its curve at the reference threshold."""
    tau, p0 = HeuristicPredictor().curve(reps)
    return invert_curve(tau, H.reference_threshold, math.inf, p0)


def test_empty_history_raises():
    with pytest.raises(ValueError):
        HeuristicPredictor().curve([])


def test_every_curve_is_ceiling_free():
    assert HeuristicPredictor().curve(make_reps([(T0, False)])).ceiling == NO_CEILING


def test_failure_schedules_relearning_step():
    assert interval_of(make_reps([(T0, True), (T0 + DAY, False)])) == pytest.approx(
        H.relearn_interval
    )


def test_first_success_is_one_day():
    assert interval_of(make_reps([(T0, True)])) == pytest.approx(H.first_interval)


def test_first_success_after_lapse_restarts_the_ladder():
    reps = make_reps([(T0, True), (T0 + 10 * DAY, True), (T0 + 40 * DAY, False), (T0 + 41 * DAY, True)])
    assert interval_of(reps) == pytest.approx(H.first_interval)


def test_success_streak_multiplies_gap_by_ease():
    reps = make_reps([(T0, True), (T0 + 3 * DAY, True)])
    ease = H.ease_start + 2 * H.ease_bonus
    assert interval_of(reps) == pytest.approx(3 * DAY * ease)


def test_in_session_relearn_is_floored_at_first_interval():
    reps = make_reps([(T0, True), (T0 + 60, True)])
    assert interval_of(reps) == pytest.approx(H.first_interval)


def test_interval_is_capped():
    reps = make_reps([(T0, True), (T0 + 1000 * DAY, True)])
    assert interval_of(reps) == pytest.approx(H.max_interval)


def test_ease_is_bounded():
    p = HeuristicPredictor()
    many_failures = make_reps([(T0 + i, False) for i in range(50)])
    many_successes = make_reps([(T0 + i, True) for i in range(50)])
    assert p._ease(many_failures) == H.ease_min
    assert p._ease(many_successes) == H.ease_max


def test_next_repetition_delta_at_reference_threshold_is_the_interval():
    p = HeuristicPredictor(PredictConfig(recall_threshold=H.reference_threshold, max_delta_seconds=math.inf))
    assert p.next_repetition_delta(make_reps([(T0, True)])) == pytest.approx(H.first_interval)


def test_higher_user_threshold_shortens_interval():
    reps = make_reps([(T0, True), (T0 + 3 * DAY, True)])
    lax = HeuristicPredictor(PredictConfig(recall_threshold=0.7))
    strict = HeuristicPredictor(PredictConfig(recall_threshold=0.9))
    assert strict.next_repetition_delta(reps) < lax.next_repetition_delta(reps)


def test_recall_probability_at_interval_is_reference_threshold():
    reps = make_reps([(T0, True)])
    assert HeuristicPredictor().recall_probability(reps, H.first_interval) == pytest.approx(
        H.reference_threshold
    )


def test_post_rep_curves_appends_both_outcomes():
    p = HeuristicPredictor()
    reps = make_reps([(T0, True), (T0 + 3 * DAY, True)])
    (ok, no), (new_ok, new_no) = p.post_rep_curves(
        [(reps, Direction.FORWARD), ([], Direction.REVERSE)], practiced_at=T0 + 10 * DAY
    )
    assert ok.tau > no.tau
    assert no.tau == pytest.approx(H.relearn_interval / math.log(1 / H.reference_threshold))
    # A never-practised card: the hypothetical rep is its whole history.
    assert new_ok.tau == pytest.approx(H.first_interval / math.log(1 / H.reference_threshold))
    assert new_no.tau == no.tau
