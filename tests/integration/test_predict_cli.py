"""Smoke test for ``python -m src.predict`` without a checkpoint (heuristic fallback)."""

import sys

import pytest

from src import predict
from src.database import Direction
from tests.factories import DAY, T0, add_word


@pytest.fixture
def run(db, tmp_path, monkeypatch, capsys):
    # The CLI resolves the checkpoint relative to cwd; an empty cwd forces the heuristic.
    monkeypatch.chdir(tmp_path)

    def _run(*args: str) -> str:
        monkeypatch.setattr(sys, "argv", ["predict", "--db", str(db), *args])
        predict.main()
        return capsys.readouterr().out

    return _run


def test_word_lookup_with_history(run):
    add_word(
        "chat",
        "kot",
        {Direction.FORWARD: [(T0, True), (T0 + DAY, True), (T0 + 4 * DAY, False)]},
    )
    out = run("--word", "cha", "--history")
    assert "heuristic schedule" in out
    assert "chat → kot" in out
    assert "P(recall now)" in out
    assert "History" in out


def test_unknown_word_id(run):
    assert "no word with id 999" in run("--word-id", "999")


def test_no_match(run):
    assert "No words matching" in run("--word", "zzz")


def test_fmt_duration():
    assert predict._fmt_duration(2 * DAY + 4 * 3600 + 13 * 60) == "2d 4h 13m"
    assert predict._fmt_duration(30) == "0m"
