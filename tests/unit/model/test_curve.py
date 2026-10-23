"""Forgetting-curve math: activation, evaluation, inversion and the gain integrals."""
import math

import pytest
import torch

from src.model.curve import (
    CEILING_LOGIT_INIT,
    CEILING_LOGIT_MAX,
    LOG_TAU_INIT,
    LOG_TAU_MAX,
    NO_CEILING,
    PARAM_EPS,
    Curve,
    ceiling,
    curve_from,
    curve_recall,
    expected_gain,
    expected_retained,
    invert_curve,
    next_delta,
    recall_at,
    remaining_seconds,
    retained_seconds,
    time_constant,
)

DAY = 86_400.0


def raw(log_tau: float, logit: float) -> torch.Tensor:
    return torch.tensor([log_tau, logit])


class TestActivation:
    def test_time_constant_is_exp_of_channel_zero(self):
        assert time_constant(raw(math.log(DAY), 0.0)).item() == pytest.approx(DAY, rel=1e-5)

    def test_time_constant_clamps_extreme_raw_output(self):
        assert math.isfinite(time_constant(raw(1e6, 0.0)).item())
        assert time_constant(raw(1e6, 0.0)).item() == pytest.approx(math.exp(LOG_TAU_MAX), rel=1e-5)
        assert time_constant(raw(-1e6, 0.0)).item() > 0.0

    def test_ceiling_is_sigmoid_of_channel_one(self):
        assert ceiling(raw(0.0, 0.0)).item() == pytest.approx(0.5)
        assert ceiling(raw(0.0, CEILING_LOGIT_INIT)).item() == pytest.approx(0.98, abs=1e-6)

    def test_ceiling_stays_strictly_inside_unit_interval(self):
        assert 0.0 < ceiling(raw(0.0, -1e6)).item()
        assert ceiling(raw(0.0, 1e6)).item() <= 1.0

    def test_curve_from_returns_plain_floats(self):
        curve = curve_from(raw(LOG_TAU_INIT, 0.0))
        assert isinstance(curve.tau, float) and isinstance(curve.ceiling, float)
        assert curve.tau == pytest.approx(3 * DAY, rel=1e-5)
        assert curve.ceiling == pytest.approx(0.5)


class TestRecall:
    def test_recall_at_zero_gap_is_the_ceiling(self):
        assert recall_at(DAY, 0.0, 0.7) == pytest.approx(0.7)

    def test_recall_at_one_time_constant_is_ceiling_over_e(self):
        assert recall_at(DAY, DAY, 0.7) == pytest.approx(0.7 / math.e)

    def test_recall_at_defaults_to_no_ceiling(self):
        assert recall_at(DAY, DAY) == pytest.approx(math.exp(-1))

    def test_recall_at_clamps_negative_gap_and_bounds(self):
        assert recall_at(DAY, -100.0, 0.5) == pytest.approx(0.5)
        assert recall_at(DAY, 0.0, NO_CEILING) == pytest.approx(1.0 - PARAM_EPS)
        assert recall_at(1.0, 1e9) == pytest.approx(PARAM_EPS)

    def test_recall_is_monotonically_decreasing(self):
        values = [recall_at(DAY, d, 0.9) for d in range(0, 10 * int(DAY), int(DAY))]
        assert values == sorted(values, reverse=True)

    def test_tensor_and_float_versions_agree(self):
        r = raw(math.log(2 * DAY), 1.5).view(1, 1, 2)
        deltas = torch.tensor([[DAY]])
        p0 = 1 / (1 + math.exp(-1.5))
        assert curve_recall(deltas, r).item() == pytest.approx(
            recall_at(2 * DAY, DAY, p0), rel=1e-5
        )

    def test_curve_recall_shape(self):
        r = torch.zeros(3, 4, 2)
        assert curve_recall(torch.ones(3, 4), r).shape == (3, 4)


class TestInversion:
    def test_inverts_recall_at(self):
        tau, p0, theta = 5 * DAY, 0.9, 0.6
        delta = invert_curve(tau, theta, math.inf, p0)
        assert recall_at(tau, delta, p0) == pytest.approx(theta)

    def test_closed_form(self):
        assert invert_curve(DAY, 0.8) == pytest.approx(DAY * math.log(1 / 0.8))

    def test_unreachable_threshold_is_due_now(self):
        assert invert_curve(DAY, 0.8, p0=0.5) == 0.0
        assert invert_curve(DAY, 0.8, p0=0.8) == 0.0

    def test_zero_threshold_returns_cap(self):
        assert invert_curve(DAY, 0.0, max_delta_seconds=123.0) == 123.0

    def test_cap_applies(self):
        assert invert_curve(1e12, 0.8, max_delta_seconds=DAY) == DAY

    def test_infinite_cap_leaves_interval_uncapped(self):
        assert invert_curve(1e12, 0.5, math.inf) == pytest.approx(1e12 * math.log(2))

    def test_higher_threshold_means_shorter_interval(self):
        assert invert_curve(DAY, 0.9) < invert_curve(DAY, 0.7)

    def test_next_delta_uses_the_emitted_ceiling(self):
        r = raw(math.log(DAY), 2.0)
        p0 = 1 / (1 + math.exp(-2.0))
        assert next_delta(r, 0.5) == pytest.approx(DAY * math.log(p0 / 0.5), rel=1e-5)


class TestIntegrals:
    def test_retained_seconds_is_ceiling_times_tau(self):
        assert retained_seconds(Curve(DAY, 0.8)) == pytest.approx(0.8 * DAY)

    def test_retained_seconds_matches_numeric_integral(self):
        tau, p0 = 2.0, 0.7
        dt = 1e-3
        numeric = sum(p0 * math.exp(-i * dt / tau) * dt for i in range(int(40 / dt)))
        assert retained_seconds(Curve(tau, p0)) == pytest.approx(numeric, rel=1e-3)

    def test_remaining_seconds_has_no_ceiling_factor(self):
        assert remaining_seconds(0.4, DAY) == pytest.approx(0.4 * DAY)

    def test_remaining_seconds_matches_tail_integral(self):
        tau, p0, s = 3.0, 0.6, 2.0
        dt = 1e-3
        tail = sum(p0 * math.exp(-(s + i * dt) / tau) * dt for i in range(int(60 / dt)))
        assert remaining_seconds(recall_at(tau, s, p0), tau) == pytest.approx(tail, rel=1e-3)

    def test_expected_retained_weights_branches_by_recall(self):
        ok, no = Curve(10 * DAY, 0.9), Curve(DAY, 0.4)
        assert expected_retained(0.75, ok, no) == pytest.approx(
            0.75 * 0.9 * 10 * DAY + 0.25 * 0.4 * DAY
        )

    def test_expected_gain_formula(self):
        ok, no = Curve(10 * DAY, 0.9), Curve(DAY, 0.4)
        p, tau_now = 0.6, 4 * DAY
        expected = p * 0.9 * 10 * DAY + (1 - p) * 0.4 * DAY - p * tau_now
        assert expected_gain(p, tau_now, ok, no) == pytest.approx(expected)

    def test_expected_gain_of_new_card_has_no_baseline(self):
        ok, no = Curve(10 * DAY, 0.9), Curve(DAY, 0.4)
        assert expected_gain(0.5, None, ok, no) == expected_retained(0.5, ok, no)

    def test_expected_gain_can_be_negative(self):
        # A long-lived card at high recall, risking a failure curve that is far shorter.
        assert expected_gain(0.95, 100 * DAY, Curve(101 * DAY, 0.95), Curve(DAY, 0.3)) < 0
