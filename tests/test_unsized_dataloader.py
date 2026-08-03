"""Tests for dataloaders whose length is unknown (streaming ``IterableDataset``).

Paddle raises ``ValueError`` from ``DataLoader.__len__`` for an
``IterableDataset``. Such a loader has no epoch length, so batch limits and the
validation schedule must fall back to "run until exhausted" rather than
crashing or inventing a batch count.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paddle
import pytest

import ocean
from ocean.utils import MisconfigurationException
from ocean.utils.data import has_len, sized_len


class _Stream(paddle.io.IterableDataset):
    """Unsized dataset yielding ``n`` samples."""

    def __init__(self, n=40):
        super().__init__()
        self.n = n

    def __iter__(self):
        for _ in range(self.n):
            yield paddle.randn([4]), paddle.randint(0, 2, [1]).squeeze()


class CountingModel(ocean.Model):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(4, 2)
        self.train_steps = 0
        self.val_steps = 0

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        self.train_steps += 1
        x, y = batch
        return paddle.nn.functional.cross_entropy(self(x), y)

    def validation_step(self, batch, batch_idx):
        self.val_steps += 1
        x, y = batch
        self.log("val_loss", paddle.nn.functional.cross_entropy(self(x), y))

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())


def _stream_loader(n=40, bs=4):
    """40 samples / batch 4 = 10 batches, with no ``__len__``."""
    return paddle.io.DataLoader(_Stream(n), batch_size=bs)


def _sized_loader(n=40, bs=4):
    ds = paddle.io.TensorDataset([paddle.randn([n, 4]), paddle.randint(0, 2, [n])])
    return paddle.io.DataLoader(ds, batch_size=bs)


def _trainer(**kw):
    kw.setdefault("verbose", 0)
    kw.setdefault("enable_progress_bar", False)
    kw.setdefault("logger", False)
    kw.setdefault("enable_checkpointing", False)
    # Keep val_steps counting only real validation runs, not the sanity check.
    kw.setdefault("num_sanity_val_steps", 0)
    return ocean.Trainer(**kw)


# --- length probing ---------------------------------------------------------


def test_sized_len_handles_paddle_value_error():
    """Paddle raises ValueError (not TypeError) for IterableDataset loaders."""
    with pytest.raises(ValueError):
        len(_stream_loader())

    assert sized_len(_stream_loader()) is None
    assert has_len(_stream_loader()) is False
    assert sized_len(_sized_loader()) == 10
    assert has_len(_sized_loader()) is True


# --- training over an unsized loader ---------------------------------------


def test_fit_over_unsized_loader_runs_to_exhaustion():
    """Training must not crash and must consume every batch each epoch."""
    model = CountingModel()
    trainer = _trainer(max_epochs=3)
    trainer.fit(model, train_dataloaders=_stream_loader())

    assert model.train_steps == 30
    assert trainer.optimizer_step == 30


def test_unsized_loader_reports_infinite_batches():
    model = CountingModel()
    trainer = _trainer(max_epochs=1)
    trainer.fit(model, train_dataloaders=_stream_loader())
    assert trainer.num_training_batches == float("inf")


def test_int_limit_still_caps_unsized_loader():
    model = CountingModel()
    trainer = _trainer(max_epochs=1, limit_train_batches=3)
    trainer.fit(model, train_dataloaders=_stream_loader())
    assert model.train_steps == 3


def test_fractional_limit_on_unsized_loader_is_rejected():
    """There is no total to take a fraction of — fail loudly, not silently."""
    trainer = _trainer(max_epochs=1, limit_train_batches=0.5)
    with pytest.raises(MisconfigurationException, match="IterableDataset"):
        trainer.fit(CountingModel(), train_dataloaders=_stream_loader())


def test_fractional_val_check_interval_on_unsized_loader_is_rejected():
    trainer = _trainer(max_epochs=1, val_check_interval=0.5)
    with pytest.raises(MisconfigurationException, match="val_check_interval"):
        trainer.fit(
            CountingModel(),
            train_dataloaders=_stream_loader(),
            val_dataloaders=_sized_loader(),
        )


def test_int_val_check_interval_works_on_unsized_loader():
    """An int interval is a batch count, which needs no epoch length."""
    model = CountingModel()
    trainer = _trainer(max_epochs=1, val_check_interval=5, limit_val_batches=2)
    trainer.fit(model, train_dataloaders=_stream_loader(), val_dataloaders=_sized_loader())

    assert model.train_steps == 10
    assert model.val_steps == 2 * 2  # validated twice (after batch 5 and 10)


def test_unsized_val_loader_runs_fully():
    model = CountingModel()
    trainer = _trainer(max_epochs=1, limit_train_batches=2)
    trainer.fit(model, train_dataloaders=_sized_loader(), val_dataloaders=_stream_loader())
    assert model.val_steps == 10


# --- trailing accumulation window ------------------------------------------


def test_partial_accumulation_window_is_applied_on_unsized_loader():
    """10 batches with accumulate=3 → 3 full windows + 1 trailing partial one.

    Without flushing the leftover window the last batch's gradients would be
    silently discarded, since an unsized epoch never reports a last batch.
    """
    model = CountingModel()
    trainer = _trainer(max_epochs=1, accumulate_grad_batches=3)
    trainer.fit(model, train_dataloaders=_stream_loader())

    assert model.train_steps == 10
    assert trainer.optimizer_step == 4


def test_accumulation_actually_updates_weights_on_trailing_window():
    """The flushed window must change the weights, not just bump a counter."""

    class Probe(CountingModel):
        def on_train_batch_end(self, outputs, batch, batch_idx):
            if batch_idx == 9:  # after the final batch, before the flush
                self.w_before_flush = self.linear.weight.clone()

    model = Probe()
    trainer = _trainer(max_epochs=1, accumulate_grad_batches=3)
    trainer.fit(model, train_dataloaders=_stream_loader())

    delta = (model.linear.weight - model.w_before_flush).abs().max().item()
    assert delta > 0, "trailing accumulation window never reached the optimizer"


def test_sized_loader_accumulation_unchanged():
    """The sized path keeps closing its window on the last batch."""
    model = CountingModel()
    trainer = _trainer(max_epochs=1, accumulate_grad_batches=3)
    trainer.fit(model, train_dataloaders=_sized_loader())

    assert model.train_steps == 10
    assert trainer.optimizer_step == 4
