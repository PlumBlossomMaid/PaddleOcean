"""Tests for the merged training-loop architecture.

Covers the behaviours that the loop refactor is responsible for:

* the epoch loop (``trainer.fit_loop.epoch_loop``) is the live batch loop,
* gradient accumulation is driven by batch progress (survives epoch boundaries,
  forces a step on the last batch, no separate end-of-epoch flush),
* manual optimization does not fake the optimizer-step counter,
* the loop-state checkpoint round-trips through the epoch loop, and the legacy
  (pre-refactor) checkpoint schema still loads,
* an LR scheduler that is not bound to its optimizer is flagged, and Paddle's
  ``ReduceOnPlateau`` steps with a metric.
"""

from __future__ import annotations

import warnings

import paddle
import paddle.nn as nn

import ocean


class _LinearModel(ocean.Model):
    """Minimal automatic-optimization model."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 2)
        self.opt_steps = 0

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.net.parameters())

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self.net(x).mean()

    def on_before_optimizer_step(self, optimizer):
        self.opt_steps += 1


def _loader(num_samples: int = 40, batch_size: int = 10):
    return paddle.io.DataLoader(
        paddle.io.TensorDataset([paddle.randn([num_samples, 4])]),
        batch_size=batch_size,
    )


def _trainer(tmp_path, **kwargs):
    defaults = dict(
        default_root_dir=str(tmp_path),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        limit_val_batches=0,
    )
    defaults.update(kwargs)
    return ocean.Trainer(**defaults)


# ====================================================================
# F1 - the dead epoch loop is now the live one
# ====================================================================


def test_epoch_loop_is_used(tmp_path):
    """The batch loop runs inside trainer.fit_loop.epoch_loop, not inlined."""
    model = _LinearModel()
    trainer = _trainer(tmp_path, max_epochs=1, limit_train_batches=3)
    trainer.fit(model, _loader(30, 10))

    epoch_loop = trainer.fit_loop.epoch_loop
    # The epoch loop's batch progress advanced -> it actually drove training.
    assert epoch_loop.batch_progress.total.completed == 3
    assert trainer.optimizer_step == 3


# ====================================================================
# F5 - accumulation via progress counter, forced step on last batch
# ====================================================================


def test_accumulation_even_division(tmp_path):
    """4 batches, accumulate=2 -> 2 optimizer steps."""
    model = _LinearModel()
    trainer = _trainer(tmp_path, max_epochs=1, limit_train_batches=4, accumulate_grad_batches=2)
    trainer.fit(model, _loader(40, 10))
    assert trainer.optimizer_step == 2
    assert model.opt_steps == 2


def test_accumulation_forces_step_on_last_batch(tmp_path):
    """3 batches, accumulate=2 -> full window (1) + forced last-batch step (1) = 2.

    The old code relied on a per-epoch flush block; the merged loop steps on the
    final batch instead, so there is no leftover-gradient double step.
    """
    model = _LinearModel()
    trainer = _trainer(tmp_path, max_epochs=1, limit_train_batches=3, accumulate_grad_batches=2)
    trainer.fit(model, _loader(30, 10))
    assert trainer.optimizer_step == 2
    assert model.opt_steps == 2


def test_accumulation_across_two_epochs(tmp_path):
    """Accumulation restarts each epoch: 3 batches x 2 epochs, accumulate=2 -> 4 steps.

    Because the window is keyed on per-epoch batch progress (not a counter that
    leaks across epochs), each epoch independently does 1 full window + 1 forced
    last-batch step.
    """
    model = _LinearModel()
    trainer = _trainer(tmp_path, max_epochs=2, limit_train_batches=3, accumulate_grad_batches=2)
    trainer.fit(model, _loader(30, 10))
    assert trainer.optimizer_step == 4


# ====================================================================
# F4 - manual optimization does not fake the optimizer-step counter
# ====================================================================


class _ManualModel(ocean.Model):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 2)
        self.automatic_optimization = False
        self.step_calls = 0

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.net.parameters())

    def training_step(self, batch, batch_idx):
        # User owns backward/step; deliberately does NOT call optimizer.step().
        self.step_calls += 1
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self.net(x).mean()


def test_manual_mode_does_not_fake_optimizer_step(tmp_path):
    """Manual training_step that never steps -> optimizer_step stays 0."""
    model = _ManualModel()
    trainer = _trainer(tmp_path, max_epochs=1, limit_train_batches=3)
    trainer.fit(model, _loader(30, 10))
    assert model.step_calls == 3  # training_step ran for every batch
    assert trainer.optimizer_step == 0  # but no optimizer step was faked
    assert trainer.dataloader_step == 3  # dataloader_step still advances (drives max_steps)


# ====================================================================
# Checkpoint loop-state round-trip + legacy schema
# ====================================================================


def test_loop_state_dict_round_trips_through_epoch_loop(tmp_path):
    """fit_loop.state_dict() nests batch progress under the epoch loop."""
    model = _LinearModel()
    trainer = _trainer(tmp_path, max_epochs=1, limit_train_batches=4)
    trainer.fit(model, _loader(40, 10))

    state = trainer.fit_loop.state_dict()
    assert "epoch_loop" in state
    assert state["epoch_loop"]["batch_progress"]["total"]["completed"] == 4


def test_legacy_checkpoint_schema_loads(tmp_path):
    """A pre-refactor checkpoint (batch_progress at top level) still loads."""
    trainer = _trainer(tmp_path, max_epochs=1)
    legacy = {"batch_progress": {"total": {"ready": 7, "completed": 7}, "current": {"ready": 7, "completed": 7}}}
    trainer.fit_loop.load_state_dict(legacy)
    # Routed into the epoch loop's batch progress.
    assert trainer.fit_loop.epoch_loop.batch_progress.total.completed == 7
    assert trainer.fit_loop.restarting is True


# ====================================================================
# Paddle-specific scheduler handling
# ====================================================================


def test_unbound_scheduler_warns():
    """A scheduler not passed as the optimizer's learning_rate is flagged."""
    from ocean.core.optimizer import init_optimizers_and_lr_schedulers

    class _BadModel(ocean.Model):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Linear(4, 2)

        def configure_optimizers(self):
            scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.1, step_size=1, gamma=0.5)
            # BUG: optimizer uses a float LR, scheduler is never bound to it.
            opt = paddle.optimizer.SGD(learning_rate=0.1, parameters=self.net.parameters())
            return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        init_optimizers_and_lr_schedulers(_BadModel())
    assert any("not bound" in str(w.message) for w in caught)


def test_bound_scheduler_does_not_warn():
    """A correctly bound scheduler produces no binding warning."""
    from ocean.core.optimizer import init_optimizers_and_lr_schedulers

    class _GoodModel(ocean.Model):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Linear(4, 2)

        def configure_optimizers(self):
            scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.1, step_size=1, gamma=0.5)
            opt = paddle.optimizer.SGD(learning_rate=scheduler, parameters=self.net.parameters())
            return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        init_optimizers_and_lr_schedulers(_GoodModel())
    assert not any("not bound" in str(w.message) for w in caught)


def test_reduce_on_plateau_steps_with_metric(tmp_path):
    """ReduceOnPlateau receives the monitored metric and decays without error."""

    class _PlateauModel(ocean.Model):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Linear(4, 2)

        def configure_optimizers(self):
            scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=0.1, patience=0, factor=0.5)
            opt = paddle.optimizer.SGD(learning_rate=scheduler, parameters=self.net.parameters())
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": "train_loss"},
            }

        def training_step(self, batch, batch_idx):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            loss = self.net(x).mean()
            self.log("train_loss", loss)
            return loss

    model = _PlateauModel()
    trainer = _trainer(tmp_path, max_epochs=3, limit_train_batches=2)
    # Must not raise: ReduceOnPlateau.step() requires the metric argument.
    trainer.fit(model, _loader(20, 10))
