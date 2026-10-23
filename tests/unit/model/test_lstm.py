"""The network, its checkpoint round-trip, and the batched final-step forward."""
import pytest
import torch

from src.database import Direction
from src.model.checkpoint import load_model, save_checkpoint
from src.model.curve import CEILING_INIT, HEAD_OUTPUTS, LOG_TAU_INIT, curve_from, recall_at
from src.model.features import history_rows
from src.model.inference import Predictor
from src.model.inference.batch import final_step_curves
from src.model.lstm import RecallLSTM
from tests.factories import DAY, T0, make_reps


class TestRecallLSTM:
    def test_output_shape(self, tiny_model):
        assert tiny_model(torch.zeros(3, 7, 5)).shape == (3, 7, HEAD_OUTPUTS)

    def test_head_bias_initialisation(self):
        model = RecallLSTM(hidden_size=4, num_layers=1)
        assert model.head.bias[0].item() == pytest.approx(LOG_TAU_INIT)
        assert torch.sigmoid(model.head.bias[1]).item() == pytest.approx(CEILING_INIT, abs=1e-6)

    def test_fixed_ceiling_pins_p0_to_one(self):
        torch.manual_seed(0)
        model = RecallLSTM(hidden_size=4, num_layers=1, fixed_ceiling=True).eval()
        curve = curve_from(model(torch.randn(1, 3, 5))[0, -1])
        assert curve.ceiling == pytest.approx(1.0, abs=1e-6)

    def test_is_causal(self, tiny_model):
        """Changing a later step must not change earlier outputs."""
        x = torch.randn(1, 6, 5)
        y = x.clone()
        y[0, 4:] = torch.randn(2, 5)
        with torch.no_grad():
            assert torch.allclose(tiny_model(x)[0, :4], tiny_model(y)[0, :4])

    def test_hyperparams_rebuild_same_architecture(self):
        model = RecallLSTM(hidden_size=6, num_layers=2, dropout=0.1, fixed_ceiling=True)
        assert RecallLSTM(**model.hyperparams()).hyperparams() == model.hyperparams()


class TestCheckpoint:
    def test_round_trip_preserves_predictions(self, tiny_model, tmp_path):
        path = tmp_path / "m.pt"
        save_checkpoint(tiny_model, path, val_loss=0.5, epoch=3)
        loaded = load_model(path, torch.device("cpu"))
        x = torch.randn(2, 4, 5)
        with torch.no_grad():
            assert torch.allclose(loaded(x), tiny_model(x))
        assert not loaded.training

    def test_single_channel_checkpoint_is_widened_to_no_ceiling(self, tiny_model, tmp_path):
        state = tiny_model.state_dict()
        state["head.weight"] = state["head.weight"][:1].clone()
        state["head.bias"] = state["head.bias"][:1].clone()
        path = tmp_path / "old.pt"
        torch.save(
            {"state_dict": state, "hyperparams": tiny_model.hyperparams(), "val_loss": 0.0, "epoch": 0},
            path,
        )
        loaded = load_model(path, torch.device("cpu"))
        x = torch.randn(1, 3, 5)
        with torch.no_grad():
            curve = curve_from(loaded(x)[0, -1])
            assert curve.ceiling == pytest.approx(1.0, abs=1e-6)
            assert torch.allclose(loaded(x)[..., 0], tiny_model(x)[..., 0])


class TestBatchedForward:
    def histories(self):
        return [
            history_rows(make_reps([(T0 + i * DAY, i % 2 == 0) for i in range(n)]), Direction.FORWARD)
            for n in (5, 1, 3, 8, 2)
        ]

    def test_empty(self, tiny_model):
        assert final_step_curves(tiny_model, []) == []

    @pytest.mark.parametrize("chunk_size", [None, 1, 2, 64])
    def test_matches_unbatched_forward_in_input_order(self, tiny_model, chunk_size):
        seqs = self.histories()
        got = final_step_curves(tiny_model, seqs, chunk_size=chunk_size)
        for rows, curve in zip(seqs, got):
            with torch.no_grad():
                want = curve_from(tiny_model(torch.tensor([rows]))[0, -1])
            assert curve.tau == pytest.approx(want.tau, rel=1e-4)
            assert curve.ceiling == pytest.approx(want.ceiling, rel=1e-4)


class TestPredictor:
    def test_empty_history_raises(self, tiny_model):
        with pytest.raises(ValueError):
            Predictor(tiny_model).curve([], Direction.FORWARD)

    def test_recall_probability_agrees_with_curve(self, tiny_model):
        p = Predictor(tiny_model)
        reps = make_reps([(T0, True), (T0 + DAY, True)])
        tau, p0 = p.curve(reps, Direction.FORWARD)
        assert p.recall_probability(reps, DAY, Direction.FORWARD) == pytest.approx(
            recall_at(tau, DAY, p0), rel=1e-4
        )

    def test_post_rep_curves_match_explicit_appended_history(self, tiny_model):
        p = Predictor(tiny_model)
        reps = make_reps([(T0, True), (T0 + DAY, False)])
        now = T0 + 4 * DAY
        [(ok, no)] = p.post_rep_curves([(reps, Direction.FORWARD)], now)
        for remembered, got in ((True, ok), (False, no)):
            extended = reps + make_reps([(now, remembered)])
            want = p.curve(extended, Direction.FORWARD)
            assert got.tau == pytest.approx(want.tau, rel=1e-4)
            assert got.ceiling == pytest.approx(want.ceiling, rel=1e-4)

    def test_post_rep_curves_accepts_empty_history(self, tiny_model):
        [(ok, no)] = Predictor(tiny_model).post_rep_curves([([], Direction.REVERSE)], T0)
        assert ok.tau > 0 and no.tau > 0
