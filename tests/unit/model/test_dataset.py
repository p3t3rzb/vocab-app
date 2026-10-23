"""Training sequences and the time-based train/val split."""

import math

import pytest
import torch

from src.database import Direction, Word
from src.model.dataset import Sequence, split_by_time
from tests.factories import DAY, T0, make_reps


def word(word_id: int = 1) -> Word:
    return Word(id=word_id, source_text="chat", target_text="kot")


def seq(word_id: int, times: list[int]) -> Sequence:
    reps = make_reps([(t, True) for t in times], word_id)
    s = Sequence.from_repetitions(word(word_id), Direction.FORWARD, reps)
    assert s is not None
    return s


class TestFromRepetitions:
    def test_needs_at_least_two_reps(self):
        reps = make_reps([(T0, True)])
        assert Sequence.from_repetitions(word(), Direction.FORWARD, reps) is None

    def test_inputs_carry_gap_and_previous_outcome_targets_current(self):
        reps = make_reps([(T0, False), (T0 + DAY, True), (T0 + 3 * DAY, False)])
        s = Sequence.from_repetitions(word(), Direction.REVERSE, reps)
        assert s.inputs.shape == (2, 5)
        assert s.inputs[:, 0].tolist() == pytest.approx(
            [math.log(DAY + 1), math.log(2 * DAY + 1)]
        )
        # prev_remembered / prev_not_remembered come from the *previous* rep.
        assert s.inputs[:, 1].tolist() == [0.0, 1.0]
        assert s.inputs[:, 2].tolist() == [1.0, 0.0]
        # direction one-hot: reverse.
        assert s.inputs[:, 3].tolist() == [0.0, 0.0]
        assert s.inputs[:, 4].tolist() == [1.0, 1.0]
        assert s.targets.tolist() == [1.0, 0.0]
        assert s.practiced_at.tolist() == [T0 + DAY, T0 + 3 * DAY]
        assert s.scored_from == 0


class TestSplitByTime:
    def test_one_cutoff_for_every_word(self):
        a = seq(1, [T0 + i * DAY for i in range(11)])  # 10 scored steps, days 1..10
        b = seq(2, [T0 + 8 * DAY, T0 + 9 * DAY, T0 + 10 * DAY])  # days 9, 10
        train, val = split_by_time([a, b], val_split=0.25)

        cutoff = min(int(s.practiced_at[s.scored_from]) for s in val)
        for s in train:
            assert int(s.practiced_at.max()) < cutoff
        for s in val:
            assert bool((s.practiced_at[s.scored_from :] >= cutoff).all())

    def test_val_split_fraction_and_cutoff_rep_goes_to_val(self):
        a = seq(1, [T0 + i * DAY for i in range(11)])  # scored reps on days 1..10
        train, val = split_by_time([a], val_split=0.3)
        assert len(train[0].targets) == 7
        assert val[0].practiced_at[val[0].scored_from :].tolist() == [
            T0 + d * DAY for d in (8, 9, 10)
        ]

    def test_reps_sharing_the_cutoff_timestamp_all_go_to_val(self):
        a = seq(1, [T0, T0 + DAY, T0 + 2 * DAY])
        b = seq(2, [T0, T0 + 2 * DAY])  # shares the cutoff timestamp with a
        train, val = split_by_time([a, b], val_split=0.4)
        assert all(int(s.practiced_at.max()) < T0 + 2 * DAY for s in train)
        assert sum(len(s.targets) - s.scored_from for s in val) == 2

    def test_val_keeps_full_history_as_input(self):
        a = seq(1, [T0 + i * DAY for i in range(11)])
        _, val = split_by_time([a], val_split=0.3)
        assert len(val) == 1
        assert len(val[0].inputs) == len(a.inputs)
        assert val[0].scored_from > 0

    def test_train_sequences_are_truncated_prefixes(self):
        a = seq(1, [T0 + i * DAY for i in range(11)])
        train, val = split_by_time([a], val_split=0.3)
        n = len(train[0].targets)
        assert n == val[0].scored_from
        assert torch.equal(train[0].inputs, a.inputs[:n])

    def test_word_started_after_cutoff_is_val_only(self):
        a = seq(1, [T0 + i * DAY for i in range(11)])
        late = seq(2, [T0 + 20 * DAY, T0 + 21 * DAY])
        train, val = split_by_time([a, late], val_split=0.2)
        assert all(s.word_id != 2 for s in train)
        assert any(s.word_id == 2 and s.scored_from == 0 for s in val)

    def test_every_scored_rep_lands_in_exactly_one_split(self):
        seqs = [seq(i, [T0 + j * DAY * i for j in range(5)]) for i in range(1, 5)]
        train, val = split_by_time(seqs, val_split=0.4)
        total = sum(len(s.targets) for s in seqs)
        assert (
            sum(len(s.targets) for s in train)
            + sum(len(s.targets) - s.scored_from for s in val)
            == total
        )
