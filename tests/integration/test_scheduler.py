"""Batched curve computation persisted onto ``words``: model-based and heuristic."""

import threading

import pytest

from src.database import Direction, Word, get_session
from src.database.repositories import RepetitionRepository
from src.model.checkpoint import save_checkpoint
from src.model.config import ScheduleConfig
from src.model.curve import NO_CEILING
from src.model.dataset import build_sequences
from src.model.inference import (
    HeuristicPredictor,
    Predictor,
    backfill_heuristic_params,
    compute_all_params,
)
from src.model.inference.scheduler import ParamScheduler
from tests.factories import DAY, T0, add_word

F, R = Direction.FORWARD, Direction.REVERSE
CURVE_COLUMNS = [
    f"{d}{o}_{p}"
    for d in ("fwd", "rev")
    for o in ("", "_ok", "_no")
    for p in ("tau", "ceiling")
]


@pytest.fixture
def deck(db):
    """Three words: both directions practised, forward only, never practised."""
    both = add_word(
        "chat", "kot", {F: [(T0, True), (T0 + DAY, True)], R: [(T0, False)]}
    )
    fwd = add_word("chien", "pies", {F: [(T0, False), (T0 + 60, True)]})
    new = add_word("oiseau", "ptak")
    return both, fwd, new


def load(word_id: int) -> Word:
    with get_session() as session:
        word = session.get(Word, word_id)
    assert word is not None
    return word


def history(word_id: int, direction: Direction):
    with get_session() as session:
        return RepetitionRepository(session).get_for_word(word_id, direction)


def test_param_scheduler_writes_curves_only_where_there_is_history(deck, tiny_model):
    both, fwd, new = deck
    ParamScheduler(tiny_model, ScheduleConfig(chunk_size=2, batch_sequences=3)).run()

    w = load(both)
    assert all(getattr(w, c) is not None for c in CURVE_COLUMNS)
    w = load(fwd)
    assert all(getattr(w, c) is not None for c in CURVE_COLUMNS if c.startswith("fwd"))
    assert all(getattr(w, c) is None for c in CURVE_COLUMNS if c.startswith("rev"))
    w = load(new)
    assert all(getattr(w, c) is None for c in CURVE_COLUMNS)


def test_param_scheduler_current_curve_matches_predictor(deck, tiny_model):
    both, _, _ = deck
    ParamScheduler(tiny_model, ScheduleConfig()).run()
    want = Predictor(tiny_model).curve(history(both, F), F)
    w = load(both)
    assert w.fwd_tau == pytest.approx(want.tau, rel=1e-4)
    assert w.fwd_ceiling == pytest.approx(want.ceiling, rel=1e-4)


def test_param_scheduler_reports_progress(deck, tiny_model):
    calls = []
    ParamScheduler(tiny_model, ScheduleConfig(chunk_size=2)).run(
        on_progress=lambda d, t: calls.append((d, t))
    )
    assert calls == [(2, 3), (3, 3)]


def test_param_scheduler_honours_stop_event(deck, tiny_model):
    stop = threading.Event()
    stop.set()
    ParamScheduler(tiny_model, ScheduleConfig()).run(stop_event=stop)
    assert load(deck[0]).fwd_tau is None


def test_compute_all_params_from_checkpoint(deck, tiny_model, tmp_path):
    path = tmp_path / "french_polish.pt"
    save_checkpoint(tiny_model, path, val_loss=0.0, epoch=0)
    compute_all_params(path)
    assert load(deck[0]).fwd_tau is not None


def test_backfill_heuristic_params(deck):
    both, fwd, new = deck
    assert backfill_heuristic_params() == 3

    w = load(both)
    want = HeuristicPredictor().curve(history(both, F))
    assert w.fwd_tau == pytest.approx(want.tau)
    assert all(
        getattr(w, c) == NO_CEILING for c in CURVE_COLUMNS if c.endswith("ceiling")
    )
    assert w.fwd_ok_tau > w.fwd_no_tau

    assert load(fwd).rev_tau is None
    assert all(getattr(load(new), c) is None for c in CURVE_COLUMNS)


def test_backfill_overwrites_a_models_ceilings(deck, tiny_model):
    ParamScheduler(tiny_model, ScheduleConfig()).run()
    assert load(deck[0]).fwd_ceiling != NO_CEILING
    backfill_heuristic_params()
    assert load(deck[0]).fwd_ceiling == NO_CEILING


def test_build_sequences_skips_single_rep_directions(deck):
    both, fwd, _ = deck
    seqs = build_sequences()
    assert {(s.word_id, s.direction) for s in seqs} == {(both, F), (fwd, F)}
    assert all(len(s.targets) == 1 for s in seqs)
