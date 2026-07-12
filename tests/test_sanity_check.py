"""Test sanity check hook sequence — aligned with Lightning behavior.

Lightning's ``_run_sanity_check`` calls the full ``_EvaluationLoop.run()``,
which fires::
    on_sanity_check_start
    → on_validation_start → validation_step × N → on_validation_end
    → on_sanity_check_end

Ocean's inline ``_sanity_check`` must match this exact sequence so that the
TQDMProgressBar properly creates and closes its sanity-progress bar and
callbacks can rely on the ``on_validation_end`` hook firing.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paddle

import ocean
from ocean.callbacks.callback import Callback


class _HookRecorder(Callback):
    """Records each hook call for later assertion."""

    def __init__(self):
        self.hooks = []

    def on_sanity_check_start(self, trainer, model):
        self.hooks.append("on_sanity_check_start")

    def on_validation_start(self, trainer, model):
        self.hooks.append("on_validation_start")

    def on_validation_batch_start(self, trainer, model, batch, batch_idx, dataloader_idx=0):
        self.hooks.append(f"on_validation_batch_start[{batch_idx}]")

    def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
        self.hooks.append(f"on_validation_batch_end[{batch_idx}]")

    def on_validation_epoch_end(self, trainer, model):
        self.hooks.append("on_validation_epoch_end")

    def on_validation_end(self, trainer, model):
        self.hooks.append("on_validation_end")

    def on_sanity_check_end(self, trainer, model):
        self.hooks.append("on_sanity_check_end")


class _Model(ocean.Model):
    def __init__(self, val_support: bool = True):
        super().__init__()
        self.linear = paddle.nn.Linear(4, 2)
        self._val_support = val_support

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        return paddle.nn.functional.cross_entropy(self(batch[0]), batch[1])

    def validation_step(self, batch, batch_idx):
        return paddle.nn.functional.cross_entropy(self(batch[0]), batch[1])

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())


def _make_data():
    """Create small train/val datasets."""
    xs = paddle.randn([10, 4])
    ys = paddle.randint(0, 2, [10])
    train = paddle.io.TensorDataset([xs, ys])
    val = paddle.io.TensorDataset([paddle.randn([4, 4]), paddle.randint(0, 2, [4])])
    return train, val


# ------------------------------------------------------------------
# Test: sanity check fires on_validation_end
# ------------------------------------------------------------------


def test_sanity_check_fires_validation_end():
    """Sanity check must fire ``on_validation_end`` (newly added hook call).

    Old code (before fix) skipped ``on_validation_end`` during sanity check,
    leaving the TQDMProgressBar val bar visible. This is an ablation test:
    reverting the ``_sanity_check`` hook addition will make this fail.
    """
    train, val = _make_data()
    recorder = _HookRecorder()

    trainer = ocean.Trainer(
        max_epochs=1,
        num_sanity_val_steps=2,
        callbacks=[recorder],
        enable_progress_bar=False,
        logger=False,
    )
    model = _Model()
    trainer.fit(model, train_dataloaders=train, val_dataloaders=val)

    hooks = recorder.hooks
    sanity_hooks = [h for h in hooks if h.startswith("on_sanity")]

    # Sanity start/end must bracket the validation hooks
    assert "on_sanity_check_start" in hooks, "Missing on_sanity_check_start"
    assert "on_sanity_check_end" in hooks, "Missing on_sanity_check_end"

    # The fix: on_validation_end MUST be called during sanity check
    assert "on_validation_end" in hooks, (
        "Missing on_validation_end during sanity check — the _sanity_check fix is not in place"
    )

    # Sequence: sanity_start → val_start → val_batch* → val_end → sanity_end
    sanity_start_idx = hooks.index("on_sanity_check_start")
    val_start_idx = hooks.index("on_validation_start")
    val_end_idx = hooks.index("on_validation_end")
    sanity_end_idx = hooks.index("on_sanity_check_end")

    assert sanity_start_idx < val_start_idx, "on_sanity_check_start must precede on_validation_start"
    assert val_start_idx < val_end_idx, "on_validation_start must precede on_validation_end"
    assert val_end_idx < sanity_end_idx, "on_validation_end must precede on_sanity_check_end"

    # Each sanity step fires batch hooks (filter only those between
    # sanity_check_start and sanity_check_end to exclude epoch-end val).
    sanity_start_idx = hooks.index("on_sanity_check_start")
    sanity_end_idx = hooks.index("on_sanity_check_end")
    sanity_batch_hooks = [h for h in hooks[sanity_start_idx:sanity_end_idx] if h.startswith("on_validation_batch_")]
    # 2 sanity steps × {start, end} = 4 batch hooks
    assert len(sanity_batch_hooks) == 4, f"Expected 4 sanity batch hooks, got {len(sanity_batch_hooks)}"


# ------------------------------------------------------------------
# Test: sanity check without validation dataloader (no-op)
# ------------------------------------------------------------------


def test_sanity_check_no_val_dataloader():
    """When there is no val dataloader, sanity check is skipped entirely."""
    train, _ = _make_data()
    recorder = _HookRecorder()

    trainer = ocean.Trainer(
        max_epochs=1,
        num_sanity_val_steps=2,
        callbacks=[recorder],
        enable_progress_bar=False,
        logger=False,
    )
    model = _Model()
    trainer.fit(model, train_dataloaders=train)

    hooks = recorder.hooks
    assert "on_sanity_check_start" not in hooks, "Sanity check should not run without val dataloader"
    assert "on_sanity_check_end" not in hooks, "Sanity check should not run without val dataloader"
