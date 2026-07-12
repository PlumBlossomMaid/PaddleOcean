"""T2: limit_train_batches supports int AND fractional (float) limits.

Previously the training loop's effective batch count used the raw dataloader
length, so a fractional limit (e.g. 0.5) was silently ignored and every batch
ran. It now uses the resolved limit for both int and float.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paddle
import pytest

import ocean


class _CountingModel(ocean.Model):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(4, 2)
        self.batches = 0
        self.val_batches = 0
        self.test_batches = 0

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        self.batches += 1
        return paddle.nn.functional.cross_entropy(self(batch[0]), batch[1])

    def validation_step(self, batch, batch_idx):
        self.val_batches += 1
        self.log("vl", paddle.nn.functional.cross_entropy(self(batch[0]), batch[1]), on_epoch=True)

    def test_step(self, batch, batch_idx):
        self.test_batches += 1
        self.log("tl", paddle.nn.functional.cross_entropy(self(batch[0]), batch[1]), on_epoch=True)

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())


def _dl(num_samples=40, batch_size=4):
    # 40 / 4 = 10 batches
    ds = paddle.io.TensorDataset([paddle.randn([num_samples, 4]), paddle.randint(0, 2, [num_samples])])
    return paddle.io.DataLoader(ds, batch_size=batch_size)


@pytest.mark.parametrize(
    "limit,expected",
    [
        (0.5, 5),
        (0.3, 3),
        (1.0, 10),
        (3, 3),
        (10, 10),
    ],
)
def test_limit_train_batches(limit, expected):
    model = _CountingModel()
    trainer = ocean.Trainer(
        max_epochs=1,
        limit_train_batches=limit,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_dl())
    assert model.batches == expected
    assert trainer.num_training_batches == expected


def test_num_training_batches_reflects_fraction():
    model = _CountingModel()
    trainer = ocean.Trainer(
        max_epochs=1,
        limit_train_batches=0.5,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_dl())
    assert trainer.num_training_batches == 5


# ── T3: eval loops honor limit_val_batches / limit_test_batches ──────────────


@pytest.mark.parametrize("limit,expected", [(0.5, 5), (0.2, 2), (3, 3), (1.0, 10)])
def test_standalone_validate_limit(limit, expected):
    model = _CountingModel()
    trainer = ocean.Trainer(limit_val_batches=limit, logger=False, enable_checkpointing=False)
    trainer.validate(model, dataloaders=_dl())
    assert model.val_batches == expected


@pytest.mark.parametrize("limit,expected", [(0.5, 5), (0.2, 2), (3, 3)])
def test_standalone_test_limit(limit, expected):
    model = _CountingModel()
    trainer = ocean.Trainer(limit_test_batches=limit, logger=False, enable_checkpointing=False)
    trainer.test(model, dataloaders=_dl())
    assert model.test_batches == expected


def test_midepoch_validation_limit_fraction():
    model = _CountingModel()
    trainer = ocean.Trainer(
        max_epochs=1,
        val_check_interval=2,
        limit_train_batches=2,
        limit_val_batches=0.3,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_dl(16, 4), val_dataloaders=_dl())
    assert model.val_batches == 3  # 0.3 * 10 batches


def test_limit_val_batches_zero_disables():
    model = _CountingModel()
    trainer = ocean.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_dl(16, 4), val_dataloaders=_dl())
    assert model.val_batches == 0
