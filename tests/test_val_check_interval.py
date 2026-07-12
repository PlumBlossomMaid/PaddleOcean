"""Tests for the validation trigger schedule (loops-review F3).

Covers ``val_check_interval`` in all three forms (int / float / time-based),
the ``check_val_every_n_epoch`` epoch gate, and the default (``1.0``) behavior of
validating once at the end of every epoch.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paddle

import ocean


class _CountingModel(ocean.Model):
    """Counts validation_step invocations for schedule assertions."""

    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 2)
        self.val_step_calls = 0

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = paddle.nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.val_step_calls += 1
        self.log("val_loss", paddle.nn.functional.cross_entropy(self(x), y), on_epoch=True)

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())


def _dl(num_samples, batch_size):
    ds = paddle.io.TensorDataset([
        paddle.randn([num_samples, 10]),
        paddle.randint(0, 2, [num_samples]),
    ])
    return paddle.io.DataLoader(ds, batch_size=batch_size)


def _fit(model, *, epochs=2, train=(32, 8), val=(16, 8), **trainer_kwargs):
    """4 train batches/epoch and 2 val batches by default."""
    trainer = ocean.Trainer(
        max_epochs=epochs,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        **trainer_kwargs,
    )
    trainer.fit(model, train_dataloaders=_dl(*train), val_dataloaders=_dl(*val))
    return trainer


# ── default: float 1.0 validates once at epoch end ──────────────────────────


def test_default_validates_at_epoch_end():
    """Default val_check_interval=1.0 validates once per epoch (was: never)."""
    model = _CountingModel()
    trainer = _fit(model, epochs=2)  # 2 val batches * 2 epochs = 4 step calls
    assert trainer.val_check_batch == 4  # == effective train batches/epoch
    assert model.val_step_calls == 4


def test_float_fraction_validates_multiple_times_per_epoch():
    """val_check_interval=0.5 -> validate twice per epoch."""
    model = _CountingModel()
    trainer = _fit(model, epochs=1, val_check_interval=0.5)
    assert trainer.val_check_batch == 2  # int(4 * 0.5)
    # two validations per epoch * 2 val batches = 4 step calls
    assert model.val_step_calls == 4


def test_int_interval_validates_every_n_batches():
    """val_check_interval=2 (int) -> validate every 2 training batches."""
    model = _CountingModel()
    _fit(model, epochs=1, val_check_interval=2)
    # batches 2 and 4 trigger; 2 validations * 2 val batches = 4
    assert model.val_step_calls == 4


def test_check_val_every_n_epoch_gate():
    """check_val_every_n_epoch=2 -> validate only on the 2nd (and 4th, ...) epoch."""
    model = _CountingModel()
    _fit(model, epochs=2, check_val_every_n_epoch=2)
    # only epoch index 1 validates: 1 * 2 val batches = 2
    assert model.val_step_calls == 2


def test_int_interval_exceeding_batches_raises():
    """int val_check_interval > train batches (with epoch gating) is a config error."""
    model = _CountingModel()
    try:
        _fit(model, epochs=1, val_check_interval=99)
    except ValueError:
        return
    raise AssertionError("expected ValueError for val_check_interval > training batches")


def test_check_val_every_n_epoch_none_allows_large_interval():
    """With check_val_every_n_epoch=None, an int interval may span epochs."""
    model = _CountingModel()
    # 4 batches/epoch, interval 6 -> global batch 6 (epoch1 batch2) triggers once over 2 epochs
    _fit(model, epochs=2, val_check_interval=6, check_val_every_n_epoch=None)
    assert model.val_step_calls == 2  # one trigger * 2 val batches


def test_time_based_interval_validates():
    """A zero time budget validates after every batch."""
    model = _CountingModel()
    trainer = _fit(model, epochs=1, val_check_interval="00:00:00:00")
    assert trainer._val_check_time_interval == 0.0
    assert trainer.val_check_batch is None
    # every one of 4 train batches triggers; 4 * 2 val batches = 8
    assert model.val_step_calls == 8


def test_limit_val_batches_zero_disables_validation():
    """limit_val_batches=0 must disable validation regardless of interval."""
    model = _CountingModel()
    _fit(model, epochs=2, limit_val_batches=0)
    assert model.val_step_calls == 0


def test_epoch_gate_predicate_unit():
    """_should_check_val_epoch honors val dataloaders + limit + epoch cadence."""
    trainer = ocean.Trainer(max_epochs=4, check_val_every_n_epoch=2, logger=False, enable_checkpointing=False)
    trainer.val_dataloaders = [_dl(16, 8)]
    trainer.limit_val_batches = 1.0
    trainer.current_epoch = 0
    assert trainer._should_check_val_epoch() is False  # (0+1)%2 != 0
    trainer.current_epoch = 1
    assert trainer._should_check_val_epoch() is True  # (1+1)%2 == 0
    trainer.limit_val_batches = 0
    assert trainer._should_check_val_epoch() is False  # disabled by limit
