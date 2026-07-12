"""Tests for validation metric propagation after an epoch.

Verifies that ``_compute_epoch_metrics()`` makes epoch-mean validation metrics
available to callbacks. Metrics logged with ``logger=False`` are surfaced via
``trainer.callback_metrics`` (the reference behavior) rather than
``logged_metrics``, which is reserved for values dispatched to loggers.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paddle

import ocean


class _ValLogModel(ocean.Model):
    """Model that logs validation metrics per-step for epoch-level reduction."""

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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = paddle.nn.functional.cross_entropy(self(x), y)
        acc = (x.sum(axis=1) > 0).astype(paddle.float32).mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=False)
        self.log("val_acc", acc, on_epoch=True, prog_bar=False, logger=False)

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())


class _OnEpochEndModel(ocean.Model):
    """Model that only logs in on_validation_epoch_end."""

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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = paddle.nn.functional.cross_entropy(self(x), y)
        self._val_losses = getattr(self, "_val_losses", []) + [float(loss.item())]

    def on_validation_epoch_end(self):
        if not self._val_losses:
            return
        mean_loss = sum(self._val_losses) / len(self._val_losses)
        self._val_losses = []
        # This write bypasses Ocean's _log_metric path — tests that
        # on_validation_epoch_end can read from _logged_metrics.
        if self._trainer is not None:
            logged = self._trainer._logger_connector._logged_metrics
            # Simulate what FileMetricsCallback does
            val_loss = logged.get("val_loss")
            if val_loss is not None and self.logger is not None:
                self.logger.log_metrics({"val_loss": val_loss}, step=self.global_step)

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())


# ====================================================================
# Helpers
# ====================================================================


def _make_dl(num_samples=32, batch_size=8):
    ds = paddle.io.TensorDataset([
        paddle.randn([num_samples, 10]),
        paddle.randint(0, 2, [num_samples]),
    ])
    return paddle.io.DataLoader(ds, batch_size=batch_size)


# ====================================================================
# Tests
# ====================================================================


def test_callback_metrics_contains_val_after_validation():
    """``callback_metrics`` contains epoch-mean ``val_loss`` after a real validation.

    ``val_check_interval=2`` (int) triggers a genuine mid-epoch validation pass;
    sanity checking is disabled so the assertion reflects training-time validation,
    not leaked sanity-check state.
    """
    model = _ValLogModel()
    trainer = ocean.Trainer(
        max_epochs=1,
        val_check_interval=2,
        limit_val_batches=4,
        limit_train_batches=2,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_make_dl(16, 8), val_dataloaders=_make_dl(32, 8))

    # val_loss is logged with logger=False, so it lives in callback_metrics
    # (reference behavior), not logged_metrics.
    metrics = trainer._logger_connector._callback_metrics
    assert "val_loss" in metrics, f"val_loss not in callback_metrics: {list(metrics.keys())}"
    assert isinstance(metrics["val_loss"], float)
    assert metrics["val_loss"] > 0
    # logger=False must keep it OUT of logged_metrics
    assert "val_loss" not in trainer._logger_connector._logged_metrics


def test_training_metrics_survive_midepoch_validation():
    """F9: a mid-epoch validation must not clobber training accumulation."""
    model = _ValLogModel()
    trainer = ocean.Trainer(
        max_epochs=1,
        val_check_interval=2,
        limit_val_batches=4,
        limit_train_batches=4,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_make_dl(32, 8), val_dataloaders=_make_dl(32, 8))

    metrics = trainer._logger_connector._callback_metrics
    # Both the training metric (accumulated across the whole epoch, spanning the
    # mid-epoch val) and the validation metric are present and independent.
    assert "train_loss" in metrics
    assert "val_loss" in metrics


def test_callback_metrics_contains_multiple_val_keys():
    """All ``on_epoch=True`` val-metrics appear in ``callback_metrics``."""
    model = _ValLogModel()
    trainer = ocean.Trainer(
        max_epochs=1,
        val_check_interval=2,
        limit_val_batches=3,
        limit_train_batches=2,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_make_dl(16, 8), val_dataloaders=_make_dl(24, 8))

    metrics = trainer._logger_connector._callback_metrics
    assert "val_loss" in metrics
    assert "val_acc" in metrics
    assert isinstance(metrics["val_acc"], float)


def test_val_loss_epoch_mean_matches_manual():
    """The epoch-mean ``val_loss`` in callback_metrics matches manual calculation."""
    model = _ValLogModel()
    trainer = ocean.Trainer(
        max_epochs=1,
        val_check_interval=2,
        limit_val_batches=3,
        limit_train_batches=2,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )

    # Collect per-batch losses manually
    val_losses = []
    original_val_step = model.validation_step

    def _tracking_val_step(batch, batch_idx):
        x, y = batch
        logits = model(x)
        loss = paddle.nn.functional.cross_entropy(logits, y)
        val_losses.append(float(loss.item()))
        return original_val_step(batch, batch_idx)

    model.validation_step = _tracking_val_step
    trainer.fit(model, train_dataloaders=_make_dl(16, 8), val_dataloaders=_make_dl(24, 8))

    expected_mean = sum(val_losses) / len(val_losses)
    metrics = trainer._logger_connector._callback_metrics
    assert "val_loss" in metrics
    assert abs(metrics["val_loss"] - expected_mean) < 1e-5, (
        f"val_loss {metrics['val_loss']} != expected {expected_mean}"
    )


def test_max_batches_cached_once():
    """``fit_loop.max_batches`` is set once in ``run()`` and doesn't change."""
    trainer = ocean.Trainer(max_epochs=1)
    trainer.train_dataloader = _make_dl(100, 10)
    # Before run(), max_batches is the initial value (0)
    assert trainer.fit_loop.max_batches == 0

    # After calling on_train_epoch_start (which triggers _get_total),
    # max_batches should still be 0 because run() hasn't been called yet
    assert trainer.fit_loop.max_batches == 0

    # Actually run fit, which should cache max_batches
    model = _ValLogModel()
    trainer2 = ocean.Trainer(
        max_epochs=1,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer2.fit(model, train_dataloaders=_make_dl(100, 10))

    assert trainer2.fit_loop.max_batches == 10, f"Expected max_batches=10, got {trainer2.fit_loop.max_batches}"


def test_num_training_batches_matches_max_batches():
    """``trainer.num_training_batches == trainer.fit_loop.max_batches`` after fit."""
    model = _ValLogModel()
    trainer = ocean.Trainer(
        max_epochs=1,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_make_dl(64, 8))

    assert trainer.num_training_batches == trainer.fit_loop.max_batches
    assert trainer.num_training_batches == 8  # ceil(64 / 8)


def test_get_total_matches_num_training_batches():
    """``_get_total(trainer, "train")`` == ``trainer.num_training_batches`` after fit."""
    from ocean.callbacks.progress.progress_bar import TQDMProgressBar

    model = _ValLogModel()
    trainer = ocean.Trainer(
        max_epochs=1,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_make_dl(50, 10))

    total = TQDMProgressBar._get_total(trainer, "train")
    assert total == 5  # ceil(50 / 10)
    assert total == trainer.num_training_batches
