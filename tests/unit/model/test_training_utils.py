"""Training building blocks: batching, masked loss, step mask and the history shift."""

import math

import pytest
import torch

from src.model.curve import curve_recall
from src.model.training.batching import bucket_batches
from src.model.training.loss import masked_bce
from src.model.training.trainer import _predict, _step_mask


def test_bucket_batches_cover_every_index_once():
    lengths = torch.tensor([5, 1, 9, 3, 3, 7, 2])
    batches = bucket_batches(lengths, batch_size=3, shuffle=True)
    assert sorted(torch.cat(batches).tolist()) == list(range(len(lengths)))
    assert all(len(b) <= 3 for b in batches)


def test_bucket_batches_group_similar_lengths():
    lengths = torch.tensor([10, 1, 9, 2, 8, 3])
    batches = bucket_batches(lengths, batch_size=3, shuffle=False)
    assert [sorted(lengths[b].tolist()) for b in batches] == [[1, 2, 3], [8, 9, 10]]


def test_masked_bce_ignores_masked_steps_and_sums():
    pred = torch.tensor([[0.9, 0.5, 0.01]])
    target = torch.tensor([[1.0, 0.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    assert masked_bce(pred, target, mask).item() == pytest.approx(
        -math.log(0.9) - math.log(0.5), rel=1e-5
    )


def test_step_mask():
    mask = _step_mask(torch.tensor([3, 4]), torch.tensor([0, 2]), max_l=5)
    assert mask.tolist() == [
        [True, True, True, False, False],
        [False, False, True, True, False],
    ]


def test_predict_shifts_the_gap_out_of_the_model_input():
    """The model sees step i-1's gap at step i, while the curve is evaluated at step i's gap."""
    seen = {}

    class Spy(torch.nn.Module):
        def forward(self, x):
            seen["x"] = x.clone()
            return torch.zeros(*x.shape[:2], 2)

    bx = torch.tensor(
        [[[math.log(11), 1, 0, 1, 0], [math.log(101), 0, 1, 1, 0]]], dtype=torch.float32
    )
    out = _predict(Spy(), bx)

    assert seen["x"][0, :, 0].tolist() == pytest.approx([0.0, math.log(11)])
    assert torch.equal(seen["x"][..., 1:], bx[..., 1:])
    assert out[0].tolist() == pytest.approx(
        curve_recall(torch.tensor([[10.0, 100.0]]), torch.zeros(1, 2, 2))[0].tolist(),
        rel=1e-4,
    )
