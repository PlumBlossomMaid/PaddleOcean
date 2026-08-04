"""Tests for limit_*_batches resolution and validation.

Covers:
- a limit of 0 (int or float) runs zero batches instead of disabling the cap
- invalid limits are rejected when the Trainer is built, not deep in a loop
- dataloaders without a length work at all, and honour int limits
"""

import paddle
import pytest

import ocean
from ocean.model import Model
from ocean.utils import MisconfigurationException

# ── Model / loaders ──────────────────────────────────────────────────────────


class CountingModel(Model):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 2)
        self.val_batches = []
        self.test_batches = []

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return paddle.nn.functional.cross_entropy(self(x), y)

    def validation_step(self, batch, batch_idx):
        self.val_batches.append(batch_idx)

    def test_step(self, batch, batch_idx):
        self.test_batches.append(batch_idx)

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())


class Stream(paddle.io.IterableDataset):
    """An iterable dataset — len() raises ValueError on a Paddle DataLoader."""

    def __init__(self, n=40):
        super().__init__()
        self.n = n

    def __iter__(self):
        for _ in range(self.n):
            yield paddle.randn([10]), paddle.randint(0, 2, [1]).squeeze()


def make_loader(n=64, bs=8):
    ds = paddle.io.TensorDataset([paddle.randn([n, 10]), paddle.randint(0, 2, [n])])
    return paddle.io.DataLoader(ds, batch_size=bs)


def make_stream_loader(n=40, bs=8):
    return paddle.io.DataLoader(Stream(n), batch_size=bs)


def make_trainer(**kwargs):
    kwargs.setdefault("max_epochs", 1)
    kwargs.setdefault("verbose", 0)
    kwargs.setdefault("logger", False)
    kwargs.setdefault("enable_checkpointing", False)
    kwargs.setdefault("enable_progress_bar", False)
    return ocean.Trainer(**kwargs)


# ── A limit of zero means zero batches ───────────────────────────────────────


@pytest.mark.parametrize(("limit", "expected"), [(0, 0), (0.0, 0), (0.5, 4), (4, 4), (1.0, 8)])
def test_limit_train_batches(limit, expected):
    trainer = make_trainer(limit_train_batches=limit)
    trainer.fit(CountingModel(), train_dataloaders=make_loader())
    assert trainer.dataloader_step == expected


@pytest.mark.parametrize(("limit", "expected"), [(0, 0), (0.0, 0), (2, 2), (0.5, 2), (1.0, 4)])
def test_limit_val_batches_in_standalone_validate(limit, expected):
    """validate() has no epoch-level gate to fall back on, so the cap must hold."""
    model = CountingModel()
    make_trainer(limit_val_batches=limit).validate(model, dataloaders=make_loader(32))
    assert len(model.val_batches) == expected


@pytest.mark.parametrize(("limit", "expected"), [(0, 0), (0.0, 0), (2, 2), (1.0, 4)])
def test_limit_test_batches(limit, expected):
    model = CountingModel()
    make_trainer(limit_test_batches=limit).test(model, dataloaders=make_loader(32))
    assert len(model.test_batches) == expected


def test_limit_val_batches_zero_skips_validation_during_fit():
    model = CountingModel()
    trainer = make_trainer(limit_val_batches=0.0, num_sanity_val_steps=0)
    trainer.fit(model, train_dataloaders=make_loader(), val_dataloaders=make_loader(32))
    assert model.val_batches == []


# ── Validation of the argument itself ────────────────────────────────────────


@pytest.mark.parametrize("name", ["limit_train_batches", "limit_val_batches", "limit_test_batches"])
@pytest.mark.parametrize("bad", [-1, -0.5, 1.5, 2.7, "x", True, [1]])
def test_invalid_limits_are_rejected_at_construction(name, bad):
    with pytest.raises(MisconfigurationException, match="has to be in"):
        make_trainer(**{name: bad})


def test_valid_limits_are_normalised():
    trainer = make_trainer(limit_train_batches=3.0, limit_val_batches=0.25)
    assert trainer.limit_train_batches == 3
    assert isinstance(trainer.limit_train_batches, int)
    assert trainer.limit_val_batches == 0.25


def test_fraction_rounding_to_zero_is_rejected():
    """0.01 * 8 batches < 1 would silently train nothing."""
    trainer = make_trainer(limit_train_batches=0.01)
    with pytest.raises(MisconfigurationException, match="Please increase"):
        trainer.fit(CountingModel(), train_dataloaders=make_loader())


def test_invalid_val_check_interval_is_rejected():
    with pytest.raises(MisconfigurationException, match="has to be in"):
        make_trainer(val_check_interval=-1)


# ── Dataloaders without a length ─────────────────────────────────────────────


def test_iterable_dataset_trains():
    """len() raises ValueError here; a narrower guard made fit() crash outright."""
    trainer = make_trainer()
    trainer.fit(CountingModel(), train_dataloaders=make_stream_loader(40))
    assert trainer.dataloader_step == 5


def test_iterable_dataset_honours_int_limit():
    trainer = make_trainer(limit_train_batches=3)
    trainer.fit(CountingModel(), train_dataloaders=make_stream_loader(40))
    assert trainer.dataloader_step == 3


def test_iterable_dataset_rejects_fractional_limit():
    trainer = make_trainer(limit_train_batches=0.5)
    with pytest.raises(MisconfigurationException, match="must be `1.0` or an int"):
        trainer.fit(CountingModel(), train_dataloaders=make_stream_loader(40))


def test_iterable_dataset_rejects_fractional_val_check_interval():
    trainer = make_trainer(val_check_interval=0.5)
    with pytest.raises(MisconfigurationException, match="val_check_interval"):
        trainer.fit(CountingModel(), train_dataloaders=make_stream_loader(40))
