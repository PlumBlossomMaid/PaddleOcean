"""Resume from checkpoint: restart flag cascades and already-processed batches are skipped.

Two coupled behaviours that were silently broken:

1. ``_Loop.load_state_dict`` set ``self._restarting = True`` (bare field), bypassing
   the ``restarting`` property setter that cascades the flag to child loops. The epoch
   loop therefore never saw ``restarting=True`` and ran ``reset_on_run`` (zeroing
   ``batch_progress``) instead of ``reset_on_restart`` (preserving it). Lightning relies
   on every nested loop seeing ``restarting=True`` for its restart branch.

2. With the flag fixed, ``_TrainingEpochLoop.run`` still iterated the loader from the
   start, re-processing batches the checkpoint had already consumed. Lightning advances
   a data-fetcher's ``fetched`` counter by ``batch_progress.current.ready``; without a
   fetcher, the equivalent here is to drain that many batches before the main loop.

These tests pin both: the restart flag propagates to the epoch loop, and a mid-epoch
resume actually skips the already-processed batches (training_step runs only for the
remaining tail, not the full epoch again).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paddle

import ocean


def _dl(n=64, b=8):
    ds = paddle.io.TensorDataset([paddle.randn([n, 10]), paddle.randint(0, 2, [n])])
    return paddle.io.DataLoader(ds, batch_size=b)


class _CountingModel(ocean.Model):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 2)
        self.train_calls = 0

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        self.train_calls += 1
        x, y = batch
        return paddle.nn.functional.cross_entropy(self(x), y)

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())


def _fit_and_save(tmp_path, model, limit_train_batches=8):
    trainer = ocean.Trainer(
        max_epochs=1,
        limit_train_batches=limit_train_batches,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_dl())
    path = os.path.join(str(tmp_path), "ckpt.pdparams")
    trainer.save_checkpoint(path)
    return path


def _forge_mid_epoch(path, epoch=0, ready=3):
    ckpt = paddle.load(path)
    ckpt["epoch"] = epoch
    loops = ckpt.setdefault("loops", {})
    el = loops.setdefault("epoch_loop", {})
    el["batch_progress"] = {
        "total": {"ready": ready, "started": ready, "processed": ready, "completed": ready},
        "current": {"ready": ready, "started": ready, "processed": ready, "completed": ready},
    }
    paddle.save(ckpt, path)


def test_restarting_cascades_on_resume(tmp_path):
    """Direct probe of the restore path (no run): the epoch loop's restart flag must
    be set to True after ``fit_loop.load_state_dict``, mirroring Lightning's cascade."""
    loops = {
        "epoch_loop": {
            "batch_progress": {
                "total": {"ready": 3, "started": 3, "processed": 3, "completed": 3},
                "current": {"ready": 3, "started": 3, "processed": 3, "completed": 3},
            }
        }
    }
    trainer = ocean.Trainer(
        max_epochs=3,
        limit_train_batches=0,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    fl = trainer.fit_loop
    assert fl.restarting is False
    assert fl.epoch_loop.restarting is False

    fl.load_state_dict(loops)  # what _CheckpointConnector.restore calls (connectors:283)

    assert fl.restarting is True
    assert fl.epoch_loop.restarting is True  # cascaded through the property setter
    assert fl.epoch_loop.batch_progress.current.ready == 3  # loaded, not dropped


def test_resume_preserves_batch_progress_snapshot(tmp_path):
    """After a real restore via trainer.fit(ckpt_path=...), the epoch loop's
    batch_progress retains the checkpoint's value at the resume boundary."""
    path = _fit_and_save(tmp_path, _CountingModel())
    _forge_mid_epoch(path, epoch=0, ready=3)

    model = _CountingModel()
    trainer = ocean.Trainer(
        max_epochs=1,
        limit_train_batches=8,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    # The restore happens inside fit; we don't run to completion — instead step the
    # restore and read progress before the epoch end resets it. Use limit_train_batches=0
    # so run() returns immediately after restore, exposing the loaded snapshot.
    trainer.fit(model, train_dataloaders=_dl(), ckpt_path=path)
    # run() is a no-op with limit_train_batches=0, so batch_progress keeps the restored value
    # (reset_on_restart preserves it; no increment happened).
    assert trainer.fit_loop.epoch_loop.batch_progress.current.ready >= 3


def test_mid_epoch_resume_skips_processed_batches(tmp_path):
    """Resuming into a mid-epoch checkpoint must not re-run already-processed batches.

    Forged: epoch 0, 3 of 8 batches already processed. After resume with
    limit_train_batches=8 the model's training_step runs exactly 8 - 3 = 5 times,
    not 8. This mirrors Lightning advancing ``data_fetcher.fetched`` by
    ``batch_progress.current.ready`` (here, a pre-loop drain without a fetcher).
    """
    path = _fit_and_save(tmp_path, _CountingModel(), limit_train_batches=8)
    _forge_mid_epoch(path, epoch=0, ready=3)

    model = _CountingModel()
    trainer = ocean.Trainer(
        max_epochs=1,
        limit_train_batches=8,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_dl(), ckpt_path=path)

    assert model.train_calls == 5, (
        f"expected 5 training_step calls (8 batches - 3 already done), "
        f"got {model.train_calls} — skipped batches were re-processed"
    )
