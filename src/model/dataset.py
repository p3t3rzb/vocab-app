"""Build LSTM training sequences from the repetition history in the database.

For each (word, direction) pair with ≥2 repetitions, consecutive practice
events are turned into timesteps:

* ``input  = [log(Δt + 1), prev_remembered]`` — log-time since the previous
  rep and whether that previous rep was recalled.
* ``target = remembered`` — whether the current rep was recalled.

The train/val split is done at the **word** level (not the sequence level)
so a word never leaks across the split.
"""
import math
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from src.database import Direction, RepetitionRepository, WordRepository, get_session, init_db


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


def build_sequences(db_url: str) -> list[Sequence]:
    """Load all qualifying (word, direction) sequences from the database.

    Calls :func:`init_db` (idempotently) and returns one :class:`Sequence`
    per (word, direction) pair that has at least two practice events.
    """
    init_db(db_url, "", "")
    sequences: list[Sequence] = []

    with get_session() as session:
        words = WordRepository(session).get_all()
        reps_repo = RepetitionRepository(session)

        for word in words:
            for direction in Direction:
                reps = reps_repo.get_for_word(word.id, direction)
                if len(reps) < 2:
                    continue

                inputs, targets = [], []
                for i in range(1, len(reps)):
                    delta = reps[i].practiced_at - reps[i - 1].practiced_at
                    log_delta = math.log(delta + 1)
                    prev_rem = float(reps[i - 1].remembered)
                    inputs.append([log_delta, prev_rem])
                    targets.append(float(reps[i].remembered))

                sequences.append(Sequence(
                    word_id=word.id,
                    direction=direction,
                    source_text=word.source_text,
                    target_text=word.target_text,
                    inputs=torch.tensor(inputs, dtype=torch.float32),
                    targets=torch.tensor(targets, dtype=torch.float32),
                ))

    return sequences


def split_sequences(
    sequences: list[Sequence],
    val_split: float = 0.2,
    seed: int = 42,
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


class RecallDataset(Dataset):
    """Thin :class:`~torch.utils.data.Dataset` wrapper around a list of sequences."""

    def __init__(self, sequences: list[Sequence]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Sequence:
        return self.sequences[idx]


def collate_fn(batch: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad sequences in a batch to equal length.

    Returns:
        A tuple ``(inputs, targets, lengths)`` where ``inputs`` has shape
        ``(B, max_len, 2)``, ``targets`` has shape ``(B, max_len)``, and
        ``lengths`` records the true (unpadded) length of each sequence
        so the loss can mask out padding positions.
    """
    lengths = torch.tensor([len(s.inputs) for s in batch], dtype=torch.long)
    max_len = int(lengths.max().item())

    inputs = torch.zeros(len(batch), max_len, 2)
    targets = torch.zeros(len(batch), max_len)

    for i, s in enumerate(batch):
        L = len(s.inputs)
        inputs[i, :L] = s.inputs
        targets[i, :L] = s.targets

    return inputs, targets, lengths
