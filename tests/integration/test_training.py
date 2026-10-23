"""End-to-end: train a tiny model on a synthetic deck, then schedule it."""
import random

import pytest

from src.database import Direction, Word, get_session
from src.model.checkpoint import load_model
from src.model.config import TrainConfig
from src.model.inference import compute_all_params
from src.model.training import train
from tests.factories import DAY, T0, add_word

pytestmark = pytest.mark.slow


@pytest.fixture
def synthetic_deck(db):
    """Twenty words with a few months of reps whose recall falls with the gap."""
    rng = random.Random(0)
    for i in range(20):
        history = {}
        for direction in Direction:
            t, events = T0 + rng.randrange(30) * DAY, []
            for _ in range(rng.randint(2, 8)):
                gap = rng.choice([60, 3600, DAY, 3 * DAY, 10 * DAY])
                t += gap
                events.append((t, rng.random() < 0.95 - 0.04 * gap / DAY))
            history[direction] = events
        add_word(f"w{i}", f"W{i}", history)


def test_train_then_compute_params(synthetic_deck, tmp_path):
    epochs, refit = [], []
    cfg = TrainConfig(epochs=3, hidden_size=8, num_layers=1, batch_size=8, checkpoint_dir=tmp_path)
    path = train(
        cfg,
        on_epoch=lambda e, tr, vl: epochs.append((e, tr, vl)),
        on_refit_epoch=lambda e, n, loss: refit.append((e, n)),
    )

    assert path == tmp_path / "french_polish.pt"
    assert [e for e, _, _ in epochs] == [1, 2, 3]
    assert all(tr > 0 and vl > 0 for _, tr, vl in epochs)
    assert refit and refit[-1][0] == refit[-1][1]
    assert load_model(path).hyperparams()["hidden_size"] == 8

    compute_all_params(path)
    with get_session() as session:
        words = session.query(Word).all()
    assert all(w.fwd_tau and w.rev_tau and w.fwd_ok_tau and w.rev_no_tau for w in words)
    assert all(0 < w.fwd_ceiling < 1 for w in words)
