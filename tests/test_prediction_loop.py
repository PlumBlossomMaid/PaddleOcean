"""Tests for the prediction loop.

Covers:
- the model runs in eval mode with gradients disabled, and its original
  per-sublayer training flags are restored afterwards
- limit_predict_batches is honoured (including 0 = disabled)
- return_predictions=False accumulates nothing and returns None
- module-level predict hooks fire, once per run for the epoch-level ones
- predict_step receives batch_idx / dataloader_idx when it accepts them
- trainer.predicting is True inside predict_step
"""

import os
import tempfile

import paddle

import ocean
from ocean.callbacks.prediction_writer import PredictionWriter
from ocean.model import Model

# ── Model / loaders ──────────────────────────────────────────────────────────


class PredictModel(Model):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 2)
        self.dropout = paddle.nn.Dropout(0.5)
        self.seen = []
        self.hooks = []

    def forward(self, x):
        return self.linear(self.dropout(x))

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        out = self(batch[0])
        self.seen.append((batch_idx, dataloader_idx, self.training, not out.stop_gradient))
        return out

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())

    # hooks under test
    def on_predict_start(self):
        self.hooks.append("start")

    def on_predict_epoch_start(self):
        self.hooks.append("epoch_start")

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.hooks.append(("batch_start", batch_idx, dataloader_idx))

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.hooks.append(("batch_end", batch_idx, dataloader_idx))

    def on_predict_epoch_end(self):
        self.hooks.append("epoch_end")

    def on_predict_end(self):
        self.hooks.append("end")


def make_loader(n=32, bs=8):
    ds = paddle.io.TensorDataset([paddle.randn([n, 10]), paddle.randint(0, 2, [n])])
    return paddle.io.DataLoader(ds, batch_size=bs)


def make_trainer(**kwargs):
    kwargs.setdefault("verbose", 0)
    kwargs.setdefault("logger", False)
    kwargs.setdefault("enable_checkpointing", False)
    kwargs.setdefault("enable_progress_bar", False)
    return ocean.Trainer(**kwargs)


# ── Inference context ────────────────────────────────────────────────────────


def test_predict_runs_in_eval_mode_without_grad():
    """Dropout/BatchNorm must not be in training mode, or the predictions are wrong."""
    model = PredictModel()
    model.train()
    make_trainer().predict(model, dataloaders=make_loader())

    assert model.seen, "predict_step never ran"
    for _, _, training, grad_tracked in model.seen:
        assert training is False
        assert grad_tracked is False


def test_predict_restores_previous_module_modes():
    """A sublayer deliberately left in eval mode stays that way afterwards."""
    model = PredictModel()
    model.train()
    model.dropout.eval()

    make_trainer().predict(model, dataloaders=make_loader())

    assert model.training is True
    assert model.linear.training is True
    assert model.dropout.training is False


def test_trainer_predicting_flag_is_set():
    class StageModel(PredictModel):
        def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
            self.seen.append(self._trainer.predicting)
            return self(batch[0])

    model = StageModel()
    make_trainer().predict(model, dataloaders=make_loader())
    assert all(model.seen)


# ── limit_predict_batches ────────────────────────────────────────────────────


def test_limit_predict_batches_caps_the_run():
    model = PredictModel()
    out = make_trainer(limit_predict_batches=2).predict(model, dataloaders=make_loader())
    assert len(model.seen) == 2
    assert len(out) == 2


def test_limit_predict_batches_zero_disables_prediction():
    model = PredictModel()
    out = make_trainer(limit_predict_batches=0).predict(model, dataloaders=make_loader())
    assert model.seen == []
    assert out is None


def test_fast_dev_run_bounds_prediction():
    model = PredictModel()
    make_trainer(fast_dev_run=1).predict(model, dataloaders=make_loader())
    assert len(model.seen) == 1


# ── return_predictions ───────────────────────────────────────────────────────


def test_return_predictions_false_accumulates_nothing():
    model = PredictModel()
    trainer = make_trainer()
    out = trainer.predict(model, dataloaders=make_loader(), return_predictions=False)

    assert out is None
    assert len(model.seen) == 4  # the batches still ran
    assert trainer.predict_loop._predictions == []


def test_return_predictions_false_still_feeds_an_epoch_writer():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = PredictModel()
        trainer = make_trainer(callbacks=[PredictionWriter(output_dir=tmpdir, write_interval="epoch")])
        trainer.predict(model, dataloaders=make_loader(), return_predictions=False)

        assert len(os.listdir(tmpdir)) == 4


# ── Hooks / signatures / multiple dataloaders ────────────────────────────────


def test_module_predict_hooks_fire_once_per_run():
    model = PredictModel()
    make_trainer(limit_predict_batches=2).predict(model, dataloaders=[make_loader(), make_loader()])

    assert model.hooks[0] == "start"
    assert model.hooks[1] == "epoch_start"
    assert model.hooks[-2] == "epoch_end"
    assert model.hooks[-1] == "end"
    assert model.hooks.count("epoch_start") == 1
    assert model.hooks.count("epoch_end") == 1


def test_predict_step_receives_indices():
    model = PredictModel()
    make_trainer(limit_predict_batches=2).predict(model, dataloaders=[make_loader(), make_loader()])

    assert [(b, d) for b, d, _, _ in model.seen] == [(0, 0), (1, 0), (0, 1), (1, 1)]


def test_predict_step_without_optional_params_still_works():
    """A predict_step written as (self, batch) must not be handed extra arguments."""

    class MinimalModel(Model):
        def __init__(self):
            super().__init__()
            self.linear = paddle.nn.Linear(10, 2)
            self.count = 0

        def forward(self, x):
            return self.linear(x)

        def predict_step(self, batch):
            self.count += 1
            return self(batch[0])

        def configure_optimizers(self):
            return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())

    model = MinimalModel()
    out = make_trainer().predict(model, dataloaders=make_loader())
    assert model.count == 4
    assert len(out) == 4


def test_multiple_dataloaders_return_nested_predictions():
    out = make_trainer().predict(PredictModel(), dataloaders=[make_loader(32), make_loader(16)])
    assert [len(dl_preds) for dl_preds in out] == [4, 2]


def test_single_dataloader_returns_flat_predictions():
    out = make_trainer().predict(PredictModel(), dataloaders=make_loader(32))
    assert len(out) == 4
    assert isinstance(out[0], paddle.Tensor)
