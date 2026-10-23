"""Build LSTM training sequences from the repetition history in the database.

For each (word, direction) pair with ≥2 repetitions, consecutive practice
events are turned into timesteps:

* ``input  = [log(Δt + 1), prev_remembered, prev_not_remembered, is_forward,
  is_reverse]`` — log-time since the previous rep, a one-hot encoding of
  whether that previous rep was recalled, and a one-hot encoding of the
  practice direction (constant across the sequence).
* ``target = remembered`` — whether the current rep was recalled.

The train/val split is done in **time**: every rep after one cutoff date is held
out (see :func:`split_by_time`), which is what a refit faces — learn from
everything so far, then predict what comes next.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from src.database import Direction, RepetitionRepository, WordRepository, get_session
from src.database.models import Repetition, Word

from .config import ScheduleConfig


@dataclass
class Sequence:
    """One training sequence: the repetition history of a single (word, direction) pair.

    Attributes:
        word_id: Database id of the underlying word.
        direction: Which direction (FORWARD / REVERSE) this sequence represents.
        source_text: Cached source-language text (handy for debugging).
        target_text: Cached target-language text.
        inputs: Tensor of shape ``(L, 5)`` —
            ``[log_delta, prev_remembered, prev_not_remembered, is_forward,
            is_reverse]`` per timestep.
        targets: Tensor of shape ``(L,)`` — recall outcome at each timestep.
        practiced_at: Tensor of shape ``(L,)`` — Unix time of each target rep.
        scored_from: Index of the first timestep the loss counts. Earlier steps
            are history only: the model reads them but is not scored on them.
    """

    word_id: int
    direction: Direction
    source_text: str
    target_text: str
    inputs: torch.Tensor  # (L, 5)  [log_delta, prev_rem, prev_not_rem, is_fwd, is_rev]
    targets: torch.Tensor  # (L,)    remembered at each step
    practiced_at: torch.Tensor  # (L,)  unix time of each target rep
    scored_from: int = 0

    @classmethod
    def from_repetitions(
        cls,
        word: Word,
        direction: Direction,
        reps: list[Repetition],
    ) -> Sequence | None:
        """Build a sequence from a rep history; return ``None`` if too short."""
        if len(reps) < 2:
            return None

        is_rev = float(int(direction))
        is_fwd = 1.0 - is_rev
        inputs: list[list[float]] = []
        targets: list[float] = []
        for i in range(1, len(reps)):
            delta = reps[i].practiced_at - reps[i - 1].practiced_at
            prev_rem = float(reps[i - 1].remembered)
            inputs.append(
                [math.log(delta + 1), prev_rem, 1.0 - prev_rem, is_fwd, is_rev]
            )
            targets.append(float(reps[i].remembered))

        return cls(
            word_id=word.id,
            direction=direction,
            source_text=word.source_text,
            target_text=word.target_text,
            inputs=torch.tensor(inputs, dtype=torch.float32),
            targets=torch.tensor(targets, dtype=torch.float32),
            practiced_at=torch.tensor(
                [rep.practiced_at for rep in reps[1:]], dtype=torch.int64
            ),
        )


def build_sequences() -> list[Sequence]:
    """Load all qualifying (word, direction) sequences from the active database.

    The caller is responsible for having initialized the DB via
    :func:`src.database.init_db`. Returns one :class:`Sequence` per
    (word, direction) pair that has at least two practice events.
    """
    sequences: list[Sequence] = []
    chunk_size = ScheduleConfig().chunk_size

    with get_session() as session:
        words = WordRepository(session).get_all()
        reps_repo = RepetitionRepository(session)

        # One query per chunk of words: a query per (word, direction) is ~30 s
        # on a large deck, the whole table at once is hundreds of MB of ORM rows.
        for chunk_start in range(0, len(words), chunk_size):
            chunk = words[chunk_start : chunk_start + chunk_size]
            reps_map = reps_repo.history_by_word_direction([w.id for w in chunk])
            for word in chunk:
                for direction in Direction:
                    reps = reps_map.get((word.id, int(direction)), [])
                    seq = Sequence.from_repetitions(word, direction, reps)
                    if seq is not None:
                        sequences.append(seq)

    return sequences


def split_by_time(
    sequences: list[Sequence],
    val_split: float,
) -> tuple[list[Sequence], list[Sequence]]:
    """Split sequences into ``(train, val)`` at one cutoff date shared by every word.

    The cutoff is the date after which ``val_split`` of all scored reps fall
    (reps sharing the cutoff timestamp all go to val). Train keeps each
    sequence's steps before it; val scores the steps at or after it, with the
    whole earlier history still fed in as input. Truncating a train sequence is
    the same as masking its later steps, since the LSTM is causal.

    One date for every word, rather than the last part of each word's own
    history, keeps the future out of training: a per-word split would train on
    one word's September reps while validating on another's June ones. Words
    started late have few or no steps before the cutoff, so val also tests how
    well the model handles newly-started words — as a refit has to.
    """
    times = torch.cat([s.practiced_at for s in sequences]).sort().values
    cutoff = int(times[min(int(len(times) * (1.0 - val_split)), len(times) - 1)])

    train: list[Sequence] = []
    val: list[Sequence] = []
    for s in sequences:
        n_before = int((s.practiced_at < cutoff).sum())
        if n_before > 0:
            train.append(
                replace(
                    s,
                    inputs=s.inputs[:n_before],
                    targets=s.targets[:n_before],
                    practiced_at=s.practiced_at[:n_before],
                )
            )
        if n_before < len(s.targets):
            val.append(replace(s, scored_from=n_before))
    return train, val
