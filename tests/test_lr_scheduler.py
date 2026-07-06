"""Tests for LR scheduler registration, stepping, and format parsing."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paddle

import ocean


class _StepSchedulerModel(ocean.Model):
    """Model with ``interval="step"`` scheduler."""

    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = paddle.nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.01, step_size=3, gamma=0.5)
        opt = paddle.optimizer.SGD(learning_rate=scheduler, parameters=self.parameters())
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


class _NoSchedulerModel(ocean.Model):
    """Model that returns only an optimizer."""

    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = paddle.nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())


class _EpochSchedulerModel(ocean.Model):
    """Model with ``interval="epoch"`` scheduler."""

    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = paddle.nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.01, step_size=1, gamma=0.5)
        opt = paddle.optimizer.SGD(learning_rate=scheduler, parameters=self.parameters())
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class _FlatSchedulerModel(ocean.Model):
    """Model with flat ``lr_scheduler`` key."""

    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = paddle.nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.01, step_size=3, gamma=0.5)
        opt = paddle.optimizer.SGD(learning_rate=scheduler, parameters=self.parameters())
        return {
            "optimizer": opt,
            "lr_scheduler": scheduler,
            "interval": "step",
        }


def _make_dl(num_samples=32, batch_size=8):
    ds = paddle.io.TensorDataset([
        paddle.randn([num_samples, 10]),
        paddle.randint(0, 2, [num_samples]),
    ])
    return paddle.io.DataLoader(ds, batch_size=batch_size)


# ====================================================================
# Registration tests
# ====================================================================


def test_scheduler_registered_nested_format():
    """Nested ``lr_scheduler`` dict is parsed correctly."""
    model = _StepSchedulerModel()
    trainer = ocean.Trainer(
        max_epochs=1,
        limit_val_batches=0,
        limit_train_batches=2,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_make_dl(16, 8))
    assert len(trainer._lr_schedulers) == 1
    assert trainer._lr_schedulers[0]["interval"] == "step"


def test_scheduler_registered_flat_format():
    """Flat ``lr_scheduler`` key still works."""
    model = _FlatSchedulerModel()
    trainer = ocean.Trainer(
        max_epochs=1,
        limit_val_batches=0,
        limit_train_batches=2,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_make_dl(16, 8))
    assert len(trainer._lr_schedulers) == 1
    assert trainer._lr_schedulers[0]["interval"] == "step"


def test_no_scheduler_when_configure_returns_optimizer_only():
    """No ``lr_scheduler`` config when optimizer is returned directly."""
    model = _NoSchedulerModel()
    trainer = ocean.Trainer(
        max_epochs=1,
        limit_val_batches=0,
        limit_train_batches=2,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_make_dl(16, 8))
    assert len(trainer._lr_schedulers) == 0


# ====================================================================
# LR decay tests
# ====================================================================


def test_lr_decays_with_step_interval():
    """LR decreases during training with ``interval="step"``."""
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.01, step_size=2, gamma=0.5)
    model = _StepSchedulerModel()
    lr_before = scheduler.get_lr()

    def _patched_cfg():
        return {
            "optimizer": paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters()),
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    model.configure_optimizers = _patched_cfg
    trainer = ocean.Trainer(
        max_epochs=1,
        limit_val_batches=0,
        limit_train_batches=4,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_make_dl(32, 8))
    assert scheduler.get_lr() < lr_before


def test_lr_decays_at_right_step_count():
    """LR drops by gamma after ``step_size`` optimizer steps."""
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.01, step_size=3, gamma=0.5)
    model = _StepSchedulerModel()

    def _patched_cfg():
        return {
            "optimizer": paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters()),
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    model.configure_optimizers = _patched_cfg
    trainer = ocean.Trainer(
        max_epochs=1,
        limit_val_batches=0,
        limit_train_batches=6,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_make_dl(48, 8))

    # With step_size=3 and 6 batches: 6 optimizer steps → 2 decays → LR = 0.01 * 0.5^2
    expected = 0.01 * 0.5 * 0.5
    assert abs(scheduler.get_lr() - expected) < 1e-6


# ====================================================================
# Interval behavior tests
# ====================================================================


def test_step_interval_called_multiple_times():
    """``interval="step"`` scheduler is stepped each optimizer step."""
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.01, step_size=1, gamma=0.5)
    model = _StepSchedulerModel()

    def _patched_cfg():
        return {
            "optimizer": paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters()),
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    model.configure_optimizers = _patched_cfg
    trainer = ocean.Trainer(
        max_epochs=1,
        limit_val_batches=0,
        limit_train_batches=10,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_make_dl(80, 8))
    # With step_size=1, each optimizer step decays: 10 decays total
    expected = 0.01 * (0.5**10)
    assert abs(scheduler.get_lr() - expected) < 1e-8


def test_epoch_interval_called_once_per_epoch():
    """``interval="epoch"`` scheduler is stepped once per epoch."""
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.01, step_size=1, gamma=0.5)
    model = _EpochSchedulerModel()

    def _patched_cfg():
        return {
            "optimizer": paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters()),
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    model.configure_optimizers = _patched_cfg
    trainer = ocean.Trainer(
        max_epochs=3,
        limit_val_batches=0,
        limit_train_batches=100,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_make_dl(800, 8))
    # With step_size=1: 3 epochs → 3 decays → LR = 0.01 * 0.5^3
    expected = 0.01 * (0.5**3)
    assert abs(scheduler.get_lr() - expected) < 1e-8
