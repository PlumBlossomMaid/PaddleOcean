"""Tests for the finder callbacks' state safety.

These callbacks used to corrupt the run they were attached to:

* ``LRFinder`` ramped the optimizer's learning rate across real training steps
  and never restored it, so training continued at the largest probed LR;
* ``BatchSizeFinder`` set ``trainer.limit_train_batches`` in ``on_fit_start``
  and never restored it, silently truncating the whole training run.

The real searches belong to the Tuner (``trainer.lr_find`` / ``scale_batch_size``);
these tests lock in that the callbacks no longer damage state.
"""

from __future__ import annotations

import paddle
import paddle.nn as nn

import ocean
from ocean.callbacks.batch_size_finder import BatchSizeFinder
from ocean.callbacks.lr_finder import LRFinder


class _Model(ocean.Model):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 2)

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.05, parameters=self.net.parameters())

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self.net(x).mean()


def _loader(n: int = 40, bs: int = 10):
    return paddle.io.DataLoader(paddle.io.TensorDataset([paddle.randn([n, 4])]), batch_size=bs)


def _trainer(tmp_path, **kw):
    defaults = dict(
        default_root_dir=str(tmp_path),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        limit_val_batches=0,
    )
    defaults.update(kw)
    return ocean.Trainer(**defaults)


# --------------------------------------------------------------------
# LRFinder
# --------------------------------------------------------------------
def test_lr_finder_restores_learning_rate(tmp_path):
    model = _Model()
    cb = LRFinder(min_lr=1e-4, max_lr=1.0, num_training_steps=5)
    trainer = _trainer(tmp_path, max_epochs=1, limit_train_batches=5, callbacks=[cb])
    trainer.fit(model, _loader())

    # The ramp actually ran and recorded (lr, loss) points...
    assert cb.results
    # ...but the optimizer LR is back to the configured 0.05, not the last probe.
    lr_after = float(trainer.optimizers[0]._optimizer.get_lr())
    assert abs(lr_after - 0.05) < 1e-9


def test_lr_finder_ramps_lr_during_test(tmp_path):
    model = _Model()
    cb = LRFinder(min_lr=1e-3, max_lr=0.5, num_training_steps=4)
    trainer = _trainer(tmp_path, max_epochs=1, limit_train_batches=4, callbacks=[cb])
    trainer.fit(model, _loader())
    # Recorded LRs should span a range (the ramp happened), not be constant.
    lrs = [lr for lr, _ in cb.results]
    assert max(lrs) > min(lrs)


# --------------------------------------------------------------------
# BatchSizeFinder
# --------------------------------------------------------------------
def test_batch_size_finder_does_not_truncate_training(tmp_path):
    model = _Model()
    cb = BatchSizeFinder(steps_per_trial=1)
    # 4 batches of size 10 over the 40-sample loader.
    trainer = _trainer(tmp_path, max_epochs=1, limit_train_batches=4, callbacks=[cb])
    trainer.fit(model, _loader(40, 10))

    # limit_train_batches is untouched by the callback, so the full run happened.
    assert trainer.limit_train_batches == 4
    assert trainer.fit_loop.epoch_loop.batch_progress.total.completed == 4


def test_batch_size_finder_fit_start_leaves_limits_untouched():
    cb = BatchSizeFinder(init_val=8, steps_per_trial=3)
    assert cb.optimal_batch_size == 8

    class _T:
        limit_train_batches = 1.0

    trainer = _T()
    # on_fit_start (inherited no-op) must not mutate the trainer's batch limit.
    cb.on_fit_start(trainer, None)
    assert trainer.limit_train_batches == 1.0
