"""Build LSTM training sequences from the repetition history in the database.

For each (word, direction) pair with ≥2 repetitions, consecutive practice
events are turned into timesteps:

* ``input  = [log(Δt + 1), prev_remembered]`` — log-time since the previous
  rep and whether that previous rep was recalled.
* ``target = remembered`` — whether the current rep was recalled.

The train/val split is done at the **word** level (not the sequence level)
so a word never leaks across the split.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch

from src.database import Direction, RepetitionRepository, WordRepository, get_session
from src.database.models import Repetition, Word


@dataclass
class Sequence:
    """One training sequence: the repetition history of a single (word, direction) pair.

    Attributes:
        word_id: Database id of the underlying word.
        direction: Which direction (FORWARD / REVERSE) this sequence represents.
        source_text: Cached source-language text (handy for debugging).
        target_text: Cached target-language text.
        inputs: Tensor of shape ``(L, 2)`` — ``[log_delta, prev_remembered]``
            per timestep.
        targets: Tensor of shape ``(L,)`` — recall outcome at each timestep.
    """

    word_id: int
    direction: Direction
    source_text: str
    target_text: str
    inputs: torch.Tensor   # (L, 2)  [log_delta, prev_remembered]
    targets: torch.Tensor  # (L,)    remembered at each step

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

        inputs: list[list[float]] = []
        targets: list[float] = []
        for i in range(1, len(reps)):
            delta = reps[i].practiced_at - reps[i - 1].practiced_at
            inputs.append([math.log(delta + 1), float(reps[i - 1].remembered)])
            targets.append(float(reps[i].remembered))

        return cls(
            word_id=word.id,
            direction=direction,
            source_text=word.source_text,
            target_text=word.target_text,
            inputs=torch.tensor(inputs, dtype=torch.float32),
            targets=torch.tensor(targets, dtype=torch.float32),
        )


def build_sequences() -> list[Sequence]:
    """Load all qualifying (word, direction) sequences from the active database.

    The caller is responsible for having initialized the DB via
    :func:`src.database.init_db`. Returns one :class:`Sequence` per
    (word, direction) pair that has at least two practice events.
    """
    sequences: list[Sequence] = []

    with get_session() as session:
        words = WordRepository(session).get_all()
        reps_repo = RepetitionRepository(session)

        for word in words:
            for direction in Direction:
                reps = reps_repo.get_for_word(word.id, direction)
                seq = Sequence.from_repetitions(word, direction, reps)
                if seq is not None:
                    sequences.append(seq)

    return sequences


def split_sequences(
    sequences: list[Sequence],
    val_split: float,
    seed: int,
) -> tuple[list[Sequence], list[Sequence]]:
    """Split sequences into ``(train, val)`` at the word level.

    All sequences belonging to the same word land on the same side of the
    split so the validation set is genuinely held-out — both directions of a
    word stay together.
    """
    word_ids = list({s.word_id for s in sequences})
    rng = random.Random(seed)
    rng.shuffle(word_ids)
    n_val = max(1, int(len(word_ids) * val_split))
    val_words = set(word_ids[:n_val])

    train = [s for s in sequences if s.word_id not in val_words]
    val = [s for s in sequences if s.word_id in val_words]
    return train, val
