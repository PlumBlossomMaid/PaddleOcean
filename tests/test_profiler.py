"""Tests for the profiler subsystem and its wiring into the trainer.

Covers:

* the :meth:`Profiler.profile` context manager (records timings, never crashes),
* ``PassThroughProfiler`` as the always-on default so every call site is valid,
* ``AdvancedProfiler`` inheriting ``profile``/``describe``/``setup``,
* end-to-end wiring: a fit run profiles the epoch, the batch, model hooks,
  callback hooks, the validation step and ``save_checkpoint``.
"""

from __future__ import annotations

import paddle
import paddle.nn as nn

import ocean
from ocean.callbacks import Callback
from ocean.profilers import AdvancedProfiler, PassThroughProfiler, Profiler, SimpleProfiler


# --------------------------------------------------------------------
# Unit: the profile() context manager
# --------------------------------------------------------------------
def test_profile_context_manager_records():
    """`profile` times the wrapped block and appends one record per call."""
    prof = SimpleProfiler()
    for _ in range(3):
        with prof.profile("action"):
            pass
    assert len(prof._records["action"]) == 3
    assert "Profiler Summary" in prof.summary()
    assert "action" in prof.summary()


def test_profile_never_crashes_on_stop_failure():
    """A failure while stopping is swallowed - profiling must not crash the run."""

    class _Boom(SimpleProfiler):
        def stop(self, action_name: str) -> None:  # noqa: D401
            raise RuntimeError("boom")

    prof = _Boom()
    # Should not raise despite stop() blowing up.
    with prof.profile("action"):
        pass


def test_passthrough_profiler_records_nothing():
    prof = PassThroughProfiler()
    with prof.profile("action"):
        pass
    assert prof._records == {}
    assert prof.summary() == ""


def test_advanced_profiler_inherits_profile_and_describe():
    """AdvancedProfiler is a Profiler subclass with the full context-manager API."""
    prof = AdvancedProfiler(dirpath="/tmp")
    assert isinstance(prof, Profiler)
    with prof.profile("action"):
        pass
    assert len(prof._records["action"]) == 1
    # describe()/setup() are inherited and callable without error.
    prof.setup(stage="fit", local_rank=0)
    prof.describe()


def test_describe_is_rank_zero_only():
    """A non-zero rank produces no report even when actions were recorded."""
    prof = SimpleProfiler()
    with prof.profile("action"):
        pass
    prof.setup(stage="fit", local_rank=1)
    # Not a crash, and summary() still works; describe() simply no-ops for rank>0.
    prof.describe()
    prof.setup(stage="fit", local_rank=0)
    prof.describe()


# --------------------------------------------------------------------
# Integration helpers
# --------------------------------------------------------------------
class _RecordingProfiler(SimpleProfiler):
    """SimpleProfiler that snapshots its records in ``describe`` (before teardown)."""

    def __init__(self) -> None:
        super().__init__()
        self.recorded: dict[str, int] = {}

    def describe(self) -> None:
        self.recorded = {k: len(v) for k, v in self._records.items()}
        super().describe()


class _CountingCallback(Callback):
    def __init__(self) -> None:
        self.calls = 0

    def on_train_epoch_start(self, trainer, pl_module):  # noqa: D401, ANN001
        self.calls += 1


class _Model(ocean.Model):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 2)

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.net.parameters())

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self.net(x).mean()

    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        loss = self.net(x).mean()
        self.log("val_loss", loss)
        return loss


def _loader(n: int = 20, bs: int = 10):
    return paddle.io.DataLoader(paddle.io.TensorDataset([paddle.randn([n, 4])]), batch_size=bs)


# --------------------------------------------------------------------
# Integration: default + full fit wiring
# --------------------------------------------------------------------
def test_default_profiler_is_passthrough(tmp_path):
    trainer = ocean.Trainer(default_root_dir=str(tmp_path), logger=False)
    assert isinstance(trainer.profiler, PassThroughProfiler)


def test_fit_profiles_epoch_batch_hooks_and_validation(tmp_path):
    prof = _RecordingProfiler()
    cb = _CountingCallback()
    trainer = ocean.Trainer(
        default_root_dir=str(tmp_path),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        callbacks=[cb],
        profiler=prof,
    )
    trainer.fit(_Model(), _loader(), _loader())

    recorded = prof.recorded
    assert recorded.get("run_training_epoch") == 1
    assert recorded.get("run_training_batch") == 2
    # Model hooks flow through _call_module_hook.
    assert any(k.startswith("[Model]") for k in recorded)
    # Callback hooks flow through _call_callback_hooks.
    assert any(k.startswith("[Callback]") for k in recorded)
    # The validation step is profiled during mid-epoch validation.
    assert any("validation_step" in k for k in recorded)


def test_save_checkpoint_is_profiled(tmp_path):
    prof = SimpleProfiler()
    trainer = ocean.Trainer(
        default_root_dir=str(tmp_path),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        profiler=prof,
    )
    model = _Model()
    trainer._model = model
    trainer.strategy.connect(model)
    trainer.save_checkpoint(str(tmp_path / "ckpt.pd"))
    assert "save_checkpoint" in prof._records
