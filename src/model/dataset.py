import math
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from src.database import Direction, RepetitionRepository, WordRepository, get_session, init_db


@dataclass
class Sequence:
    """One training sequence: the repetition history of a single (word, direction) pair."""
    word_id: int
    direction: Direction
    source_text: str
    target_text: str
    inputs: torch.Tensor   # (L, 2)  [log_delta, prev_remembered]
    targets: torch.Tensor  # (L,)    remembered at each step


def build_sequences(db_url: str) -> list[Sequence]:
    """Load all qualifying (word, direction) sequences from the database."""
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
                    log_delta = math.log(max(delta, 0) + 1)
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
    """Split at word level so no word leaks across train/val."""
    word_ids = list({s.word_id for s in sequences})
    rng = random.Random(seed)
    rng.shuffle(word_ids)
    n_val = max(1, int(len(word_ids) * val_split))
    val_words = set(word_ids[:n_val])

    train = [s for s in sequences if s.word_id not in val_words]
    val = [s for s in sequences if s.word_id in val_words]
    return train, val


class RecallDataset(Dataset):
    def __init__(self, sequences: list[Sequence]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Sequence:
        return self.sequences[idx]


def collate_fn(batch: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad sequences and build mask. Returns (inputs, targets, lengths)."""
    lengths = torch.tensor([len(s.inputs) for s in batch], dtype=torch.long)
    max_len = int(lengths.max().item())

    inputs = torch.zeros(len(batch), max_len, 2)
    targets = torch.zeros(len(batch), max_len)

    for i, s in enumerate(batch):
        L = len(s.inputs)
        inputs[i, :L] = s.inputs
        targets[i, :L] = s.targets

    return inputs, targets, lengths
