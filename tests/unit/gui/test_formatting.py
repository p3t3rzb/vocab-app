"""Human-readable duration and timestamp strings."""

import time
from datetime import datetime

import pytest

from src.gui.formatting import (
    day_start,
    format_due,
    format_future,
    format_past,
    format_timestamp,
)

M, H, D = 60, 3600, 86_400


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "just now"),
        (59, "just now"),
        (5 * M, "5m ago"),
        (2 * H, "2h ago"),
        (2 * H + 5 * M, "2h 5m ago"),
        (3 * D, "3d ago"),
        (2 * D + 3 * H, "2d 3h ago"),
    ],
)
def test_format_past(seconds, expected):
    assert format_past(seconds) == expected


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (-10, "due now"),
        (0, "due now"),
        (12 * M, "in 12m"),
        (H, "in 1h"),
        (H + M, "in 1h 1m"),
        (5 * D + 2 * H, "in 5d 2h"),
        (5 * D, "in 5d"),
    ],
)
def test_format_future(seconds, expected):
    assert format_future(seconds) == expected


class TestFormatDue:
    @pytest.fixture(autouse=True)
    def frozen_clock(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1_000_000.0)

    @pytest.mark.parametrize(
        "offset, expected",
        [
            (-5, "Due now"),
            (0, "Due now"),
            (30, "Due now"),  # under a minute rounds down to nothing
            (12 * M, "in 12m"),
            (2 * H, "in 2h"),
            (2 * H + 4 * M, "in 2h 4m"),
            (2 * D + 4 * H + 30 * M, "in 2d 4h"),  # minutes dropped beyond a day
            (2 * D, "in 2d"),
        ],
    )
    def test_offsets(self, offset, expected):
        assert format_due(1_000_000 + offset) == expected

    def test_none(self):
        assert format_due(None) == "–"


def test_format_timestamp():
    ts = int(datetime(2026, 3, 4, 5, 6, 7).timestamp())
    assert format_timestamp(ts) == "2026-03-04  05:06:07"


def test_day_start_is_local_midnight():
    ts = int(datetime(2026, 3, 4, 15, 30).timestamp())
    assert day_start(ts) == int(datetime(2026, 3, 4).timestamp())
    assert day_start(day_start(ts)) == day_start(ts)
