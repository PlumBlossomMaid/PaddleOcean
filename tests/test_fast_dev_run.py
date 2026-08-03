"""Tests for ``Trainer(fast_dev_run=...)``.

``fast_dev_run`` must bound the *whole run*, not just the per-epoch batch count:
one epoch, N batches, no sanity check, no logging artifacts, no checkpoints.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paddle
import pytest

import ocean
from ocean.loggers import CSVLogger, DummyLogger
from ocean.utils import MisconfigurationException


class CountingModel(ocean.Model):
    """Records how many train/val steps and epochs actually ran."""

    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 2)
        self.train_steps = 0
        self.val_steps = 0
        self.epochs = 0

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        self.train_steps += 1
        x, y = batch
        loss = paddle.nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.val_steps += 1
        x, y = batch
        self.log("val_loss", paddle.nn.functional.cross_entropy(self(x), y))

    def on_train_epoch_end(self):
        self.epochs += 1

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())


def _loader(n=64, bs=8):
    ds = paddle.io.TensorDataset([paddle.randn([n, 10]), paddle.randint(0, 2, [n])])
    return paddle.io.DataLoader(ds, batch_size=bs)


def test_fast_dev_run_stops_after_one_epoch():
    """The run must end after a single epoch of N batches, not after max_epochs."""
    model = CountingModel()
    trainer = ocean.Trainer(fast_dev_run=2, verbose=0)
    trainer.fit(model, train_dataloaders=_loader(), val_dataloaders=_loader())

    assert model.epochs == 1, f"expected exactly 1 epoch, ran {model.epochs}"
    assert model.train_steps == 2
    assert model.val_steps == 2
    assert trainer.current_epoch == 1


def test_fast_dev_run_sets_run_limits():
    """All of the run-bounding flags are derived from fast_dev_run."""
    trainer = ocean.Trainer(fast_dev_run=3, verbose=0)

    assert trainer.fast_dev_run == 3
    assert trainer.limit_train_batches == 3
    assert trainer.limit_val_batches == 3
    assert trainer.limit_test_batches == 3
    assert trainer.limit_predict_batches == 3
    assert trainer.max_epochs == 1
    assert trainer.fit_loop.max_epochs == 1
    assert trainer.max_steps == 3
    assert trainer.num_sanity_val_steps == 0
    assert trainer.val_check_interval == 1.0
    assert trainer.check_val_every_n_epoch == 1
    assert trainer._val_check_time_interval is None


def test_fast_dev_run_true_is_one_batch():
    """``fast_dev_run=True`` behaves as a single batch and stays truthy."""
    trainer = ocean.Trainer(fast_dev_run=True, verbose=0)

    assert trainer.fast_dev_run is True
    assert trainer.limit_train_batches == 1
    assert trainer.max_steps == 1

    model = CountingModel()
    trainer.fit(model, train_dataloaders=_loader(), val_dataloaders=_loader())
    assert (model.train_steps, model.val_steps, model.epochs) == (1, 1, 1)


def test_fast_dev_run_one_is_normalized_to_true():
    """``fast_dev_run=1`` reads back as ``True`` so bool checks agree."""
    assert ocean.Trainer(fast_dev_run=1, verbose=0).fast_dev_run is True


def test_fast_dev_run_negative_rejected():
    with pytest.raises(MisconfigurationException, match="should be >= 0"):
        ocean.Trainer(fast_dev_run=-1, verbose=0)


def test_fast_dev_run_disabled_leaves_flags_untouched():
    """Without fast_dev_run nothing is clamped."""
    trainer = ocean.Trainer(max_epochs=7, limit_train_batches=5, verbose=0)

    assert trainer.fast_dev_run is False
    assert trainer.max_epochs == 7
    assert trainer.limit_train_batches == 5
    assert trainer.num_sanity_val_steps == 2


def test_fast_dev_run_suppresses_logging():
    """A real logger is swapped for a no-op one; user code touching it still runs."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = ocean.Trainer(fast_dev_run=2, logger=CSVLogger(root_dir=tmp), verbose=0)
        assert len(trainer.loggers) == 1
        assert isinstance(trainer.loggers[0], DummyLogger)

        trainer.fit(CountingModel(), train_dataloaders=_loader(), val_dataloaders=_loader())

        # No metrics file was produced by the dummy.
        written = [f for _, _, files in os.walk(tmp) for f in files]
        assert written == [], f"fast_dev_run wrote log artifacts: {written}"

        # Arbitrary logger calls resolve to no-ops instead of raising.
        logger = trainer.loggers[0]
        assert logger.name == ""
        assert logger.version == ""
        assert logger[0] is logger
        assert logger.some_backend_specific_call(1, key="v") is None
        # The experiment stub swallows arbitrary backend calls (chainable no-op).
        logger.experiment.add_scalar("x", 1)


def test_fast_dev_run_without_logger_stays_empty():
    trainer = ocean.Trainer(fast_dev_run=2, logger=False, verbose=0)
    assert trainer.loggers == []


def test_fast_dev_run_suppresses_checkpointing():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = ocean.ModelCheckpoint(dirpath=tmp, save_top_k=-1, every_n_epochs=1)
        trainer = ocean.Trainer(fast_dev_run=2, callbacks=[ckpt], verbose=0)
        trainer.fit(CountingModel(), train_dataloaders=_loader(), val_dataloaders=_loader())

        written = [f for _, _, files in os.walk(tmp) for f in files]
        assert written == [], f"fast_dev_run wrote checkpoints: {written}"
