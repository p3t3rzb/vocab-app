"""Input-row construction shared by every inference path."""
import math

import pytest

from src.database import Direction
from src.model.features import history_rows, rep_row
from tests.factories import DAY, T0, make_reps


def test_rep_row_layout_forward_remembered():
    assert rep_row(DAY, True, Direction.FORWARD) == [math.log(DAY + 1), 1.0, 0.0, 1.0, 0.0]


def test_rep_row_layout_reverse_forgotten():
    assert rep_row(0.0, False, Direction.REVERSE) == [0.0, 0.0, 1.0, 0.0, 1.0]


def test_rep_row_clamps_negative_gap():
    assert rep_row(-50.0, True, Direction.FORWARD)[0] == 0.0


def test_history_rows_first_gap_is_zero_and_later_gaps_are_diffs():
    reps = make_reps([(T0, True), (T0 + DAY, False), (T0 + 3 * DAY, True)])
    rows = history_rows(reps, Direction.FORWARD)
    assert len(rows) == 3
    assert [r[0] for r in rows] == pytest.approx(
        [0.0, math.log(DAY + 1), math.log(2 * DAY + 1)]
    )
    assert [r[1] for r in rows] == [1.0, 0.0, 1.0]


def test_history_rows_empty():
    assert history_rows([], Direction.FORWARD) == []
