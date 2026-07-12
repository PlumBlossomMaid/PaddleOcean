"""C2/C3: checkpoint dump/restore symmetry and resume-past-budget guard.

C2 — state saved by dump_checkpoint() (hparams, datamodule state, and custom
state from on_save_checkpoint) must be read back by restore().
C3 — restoring an epoch beyond Trainer(max_epochs=...) must raise instead of
silently producing a no-op run.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paddle
import pytest

import ocean
from ocean.utils import MisconfigurationException


class _CustomStateModel(ocean.Model):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 2)
        self.hparams = {"lr": 0.01, "width": 10}
        self._restored_custom = None

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = paddle.nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def on_save_checkpoint(self):
        return {"custom_state": {"magic": 123}}

    def on_load_checkpoint(self, checkpoint):
        self._restored_custom = checkpoint.get("custom_state")

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())


def _dl(n=32, b=8):
    ds = paddle.io.TensorDataset([paddle.randn([n, 10]), paddle.randint(0, 2, [n])])
    return paddle.io.DataLoader(ds, batch_size=b)


def _fit_and_save(tmp_path, model, **kw):
    trainer = ocean.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        **kw,
    )
    trainer.fit(model, train_dataloaders=_dl())
    path = os.path.join(str(tmp_path), "ckpt.pdparams")
    trainer.save_checkpoint(path)
    return trainer, path


def test_hparams_restored(tmp_path):
    model = _CustomStateModel()
    _, path = _fit_and_save(tmp_path, model)

    model2 = _CustomStateModel()
    model2.hparams = {}  # wipe before resume
    trainer2 = ocean.Trainer(
        max_epochs=2,
        limit_train_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer2.fit(model2, train_dataloaders=_dl(), ckpt_path=path)
    assert model2.hparams == {"lr": 0.01, "width": 10}


def test_on_load_checkpoint_hook_called(tmp_path):
    model = _CustomStateModel()
    _, path = _fit_and_save(tmp_path, model)

    model2 = _CustomStateModel()
    assert model2._restored_custom is None
    trainer2 = ocean.Trainer(
        max_epochs=2,
        limit_train_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer2.fit(model2, train_dataloaders=_dl(), ckpt_path=path)
    assert model2._restored_custom == {"magic": 123}


def test_resume_past_max_epochs_raises(tmp_path):
    # save a checkpoint at epoch 5
    model = _CustomStateModel()
    trainer, path = _fit_and_save(tmp_path, model)
    # forge the epoch in the checkpoint to exceed a small max_epochs
    ckpt = paddle.load(path)
    ckpt["epoch"] = 5
    paddle.save(ckpt, path)

    model2 = _CustomStateModel()
    trainer2 = ocean.Trainer(max_epochs=3, num_sanity_val_steps=0, logger=False, enable_checkpointing=False)
    with pytest.raises(MisconfigurationException):
        trainer2.fit(model2, train_dataloaders=_dl(), ckpt_path=path)


def test_resume_within_max_epochs_ok(tmp_path):
    model = _CustomStateModel()
    _, path = _fit_and_save(tmp_path, model)
    ckpt = paddle.load(path)
    ckpt["epoch"] = 3
    paddle.save(ckpt, path)

    # max_epochs == restored epoch: restore succeeds (no guard trip) and the fit
    # loop is immediately done, so current_epoch stays at the restored value.
    model2 = _CustomStateModel()
    trainer2 = ocean.Trainer(
        max_epochs=3,
        limit_train_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer2.fit(model2, train_dataloaders=_dl(), ckpt_path=path)
    assert trainer2.current_epoch == 3
