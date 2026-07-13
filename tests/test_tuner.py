"""Tests for the Tuner and its mounting on the Trainer.

Covers:

* the tuner is reachable from the trainer (``trainer.tuner`` and the
  ``lr_find`` / ``scale_batch_size`` / ``tune`` entry points),
* both finders restore model + optimizer state, so tuning never corrupts the
  real weights or leaves the learning rate changed,
* ``_try_batch_size`` re-raises a non-OOM error instead of reporting it as a
  successful trial, and returns False for an OOM-like failure.
"""

from __future__ import annotations

import paddle
import paddle.nn as nn
import pytest

import ocean
from ocean.tuner import Tuner


class _Model(ocean.Model):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 2)

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.05, parameters=self.net.parameters())

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self.net(x).mean()


def _loader(n: int = 40, bs: int = 10):
    return paddle.io.DataLoader(paddle.io.TensorDataset([paddle.randn([n, 4])]), batch_size=bs)


def _trainer():
    return ocean.Trainer(logger=False, enable_progress_bar=False, enable_checkpointing=False)


def _max_weight_diff(before: dict, after: dict) -> float:
    return max(float((after[k] - before[k]).abs().max()) for k in before)


# --------------------------------------------------------------------
# Mounting
# --------------------------------------------------------------------
def test_tuner_is_mounted_on_trainer():
    trainer = _trainer()
    assert isinstance(trainer.tuner, Tuner)
    # Same instance is reused (lazy singleton).
    assert trainer.tuner is trainer.tuner


def test_trainer_exposes_tune_entry_points():
    trainer = _trainer()
    for name in ("lr_find", "scale_batch_size", "tune"):
        assert callable(getattr(trainer, name))


# --------------------------------------------------------------------
# State restoration
# --------------------------------------------------------------------
def test_lr_find_restores_weights_and_lr():
    model = _Model()
    trainer = _trainer()
    trainer._attach_model_for_tune(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    lr_before = float(trainer.optimizers[0]._optimizer.get_lr())

    suggested = trainer.lr_find(model, _loader(), num_steps=20)

    assert isinstance(suggested, float)
    assert _max_weight_diff(before, model.state_dict()) < 1e-6
    assert abs(float(trainer.optimizers[0]._optimizer.get_lr()) - lr_before) < 1e-9


def test_scale_batch_size_restores_weights_and_returns_int():
    model = _Model()
    trainer = _trainer()
    trainer._attach_model_for_tune(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}

    bs = trainer.scale_batch_size(model, _loader(), min_batch_size=2, max_batch_size=16, steps_per_trial=2)

    assert isinstance(bs, int)
    assert 2 <= bs <= 16
    assert _max_weight_diff(before, model.state_dict()) < 1e-6


def test_tune_runs_lr_find_by_default():
    model = _Model()
    trainer = _trainer()
    results = trainer.tune(model, _loader())
    assert "lr" in results
    assert "batch_size" not in results


# --------------------------------------------------------------------
# _try_batch_size error handling
# --------------------------------------------------------------------
def test_try_batch_size_reraises_non_oom_error():
    class _Bad(_Model):
        def training_step(self, batch, batch_idx):
            raise RuntimeError("deliberate model bug")

    model = _Bad()
    trainer = _trainer()
    trainer._attach_model_for_tune(model)
    tuner = Tuner(trainer)
    with pytest.raises(RuntimeError, match="deliberate model bug"):
        tuner._try_batch_size(model, [4], 4, 1, trainer._resolve_device(), paddle.randn([4, 4]))


def test_try_batch_size_returns_false_on_oom():
    class _OOM(_Model):
        def training_step(self, batch, batch_idx):
            raise RuntimeError("Out of memory on device")

    model = _OOM()
    trainer = _trainer()
    trainer._attach_model_for_tune(model)
    tuner = Tuner(trainer)
    assert tuner._try_batch_size(model, [4], 4, 1, trainer._resolve_device(), paddle.randn([4, 4])) is False
