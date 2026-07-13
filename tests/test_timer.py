"""Tests for the Timer callback's training-elapsed accounting.

The Timer measures training wall-clock as one contiguous window from
``on_train_start`` to ``on_train_end`` (with a resume offset), and tracks
validation/test stages separately. The regression these tests lock in: a
validation boundary must NOT reset the training clock (the old implementation
reset ``_training_start_time`` in ``on_validation_end``, discarding all training
time accumulated before the last validation).
"""

from __future__ import annotations

from datetime import timedelta

import paddle
import paddle.nn as nn
import pytest

import ocean
from ocean.callbacks import timer as timer_module
from ocean.callbacks.timer import Timer, _parse_duration
from ocean.trainer.states import RunningStage


class _FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _Trainer:
    """Minimal stand-in exposing the attributes Timer touches."""

    def __init__(self) -> None:
        self.should_stop = False


@pytest.fixture
def clock(monkeypatch):
    c = _FakeClock()
    monkeypatch.setattr(timer_module.time, "monotonic", c)
    return c


# --------------------------------------------------------------------
# Duration parsing
# --------------------------------------------------------------------
def test_parse_duration_formats():
    assert _parse_duration(None) is None
    assert _parse_duration(90) == 90.0
    assert _parse_duration(timedelta(minutes=2)) == 120.0
    assert _parse_duration({"minutes": 1, "seconds": 30}) == 90.0
    assert _parse_duration("01:00:00:00") == 86400.0  # DD:HH:MM:SS
    assert _parse_duration("02:00:00") == 7200.0  # HH:MM:SS


# --------------------------------------------------------------------
# Training-elapsed accounting
# --------------------------------------------------------------------
def test_training_elapsed_measures_window(clock):
    t = Timer(duration=100)
    t.on_train_start(None, None)
    clock.advance(30)
    assert t.time_elapsed(RunningStage.TRAINING) == pytest.approx(30.0)
    assert t.time_remaining() == pytest.approx(70.0)


def test_validation_boundary_does_not_reset_training_clock(clock):
    """The core regression: validation must not discard accumulated training time."""
    t = Timer(duration=100)
    t.on_train_start(None, None)  # train starts at 1000
    clock.advance(10)  # 10s of training
    t.on_validation_start(None, None)
    clock.advance(5)  # 5s of validation
    t.on_validation_end(None, None)
    clock.advance(10)  # 10s more training

    # Training window spans the whole 25s (including the mid-epoch validation),
    # and crucially is NOT reset back to ~10s by on_validation_end.
    assert t.time_elapsed(RunningStage.TRAINING) == pytest.approx(25.0)
    # Validation is tracked as its own 5s window.
    assert t.time_elapsed(RunningStage.VALIDATING) == pytest.approx(5.0)


def test_stops_when_budget_exhausted(clock):
    t = Timer(duration=50, interval="step")
    trainer = _Trainer()
    t.on_train_start(trainer, None)
    clock.advance(40)
    t.on_train_batch_end(trainer, None, None, None, 0)
    assert trainer.should_stop is False
    clock.advance(15)  # now 55s > 50s budget
    t.on_train_batch_end(trainer, None, None, None, 1)
    assert trainer.should_stop is True


def test_epoch_interval_only_checks_on_epoch_end(clock):
    t = Timer(duration=10, interval="epoch")
    trainer = _Trainer()
    t.on_train_start(trainer, None)
    clock.advance(20)
    # step-level checks are a no-op for interval="epoch"
    t.on_train_batch_end(trainer, None, None, None, 0)
    assert trainer.should_stop is False
    t.on_train_epoch_end(trainer, None)
    assert trainer.should_stop is True


# --------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------
def test_resume_offset_carries_training_time(clock):
    t = Timer(duration=100)
    t.load_state_dict({"time_elapsed": {RunningStage.TRAINING.name: 40.0}})
    t.on_train_start(None, None)
    clock.advance(10)
    # 40s carried over + 10s fresh
    assert t.time_elapsed(RunningStage.TRAINING) == pytest.approx(50.0)


def test_on_fit_start_stops_if_resumed_budget_already_spent(clock):
    t = Timer(duration=35)
    t.load_state_dict({"time_elapsed": {RunningStage.TRAINING.name: 40.0}})
    trainer = _Trainer()
    # on_fit_start runs before on_train_start; elapsed == offset == 40 >= 35.
    t.on_fit_start(trainer, None)
    assert trainer.should_stop is True


def test_state_dict_roundtrip(clock):
    t = Timer(duration=100)
    t.on_train_start(None, None)
    clock.advance(12)
    sd = t.state_dict()
    assert sd["time_elapsed"][RunningStage.TRAINING.name] == pytest.approx(12.0)
    restored = Timer(duration=100)
    restored.load_state_dict(sd)
    assert restored._offset == pytest.approx(12.0)


# --------------------------------------------------------------------
# Integration: Timer wired into a real fit
# --------------------------------------------------------------------
class _Model(ocean.Model):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 2)

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.net.parameters())

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self.net(x).mean()


def test_timer_in_real_fit_records_training_time(tmp_path):
    model = _Model()
    t = Timer(duration=3600, verbose=False)  # generous budget - should not stop
    trainer = ocean.Trainer(
        default_root_dir=str(tmp_path),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=0,
        callbacks=[t],
    )
    loader = paddle.io.DataLoader(paddle.io.TensorDataset([paddle.randn([20, 4])]), batch_size=10)
    trainer.fit(model, loader)
    # Training window was opened and closed, so elapsed is finite and >= 0.
    assert t.time_elapsed(RunningStage.TRAINING) >= 0.0
    assert t.end_time(RunningStage.TRAINING) is not None
