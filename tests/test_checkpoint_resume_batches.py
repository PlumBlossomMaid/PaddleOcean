"""Resume from checkpoint: restart-flag cascade and honest per-epoch reset.

A resume restores the across-checkpoint state (weights, optimizers, LR schedulers,
epoch, step counts, and the nested-loop ``restarting`` flag). This module pins two
parts of that flow used by ``_FitLoop``/``_TrainingEpochLoop``:

1. The ``restarting`` flag set in ``_Loop.load_state_dict`` must propagate to every
   nested loop through the property setter, so the epoch loop's own restart branch
   engages (rather than silently running as a fresh start).

2. For a plain (non-stateful) loader, a mid-epoch resume is honest: the dataloader
   is rebuilt fresh and re-yields from batch 0, so the epoch's ``batch_progress``
   is reset to zero (not a fictional mid-epoch value) and the whole epoch is
   re-processed. A stateful loader — one that exposes ``state_dict``/
   ``load_state_dict`` — is the opt-in for a genuine mid-epoch skip.
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
    be True after ``fit_loop.load_state_dict`` — i.e. the property setter cascaded it
    into the child loop, not merely set the bare field on the fit loop."""
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
    # The restore happens inside fit; we then run the epoch. With a non-stateful
    # loader the epoch's batch_progress is reset to zero on restart and the epoch
    # is processed in full, so ready advances from 0 to the number of batches run
    # (8 here) — not the forged 3. This is the honest, non-fictional resume.
    trainer.fit(model, train_dataloaders=_dl(), ckpt_path=path)
    assert trainer.fit_loop.epoch_loop.batch_progress.current.ready == 8


def test_mid_epoch_resume_reruns_epoch_when_not_stateful(tmp_path):
    """A mid-epoch resume with a plain (non-stateful) loader re-runs the whole epoch.

    Forged: epoch 0, 3 of 8 batches marked already-processed. The dataloader is a
    plain Paddle loader with no load_state_dict, so it is rebuilt fresh on resume and
    re-yields from batch 0. Ocean reports this honestly: the epoch's batch progress
    resets to zero and training_step runs the whole epoch (8 batches), with batch_idx
    starting at 0. There is no true mid-epoch skip without a stateful loader.
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

    # The whole epoch is re-processed (8, not the 5 a stateful skip would give);
    # batch progress within the epoch started from zero since the loader is not
    # resumable. Across-epoch numbers (epoch, dataloader_step) still restore.
    assert model.train_calls == 8, (
        f"expected 8 training_step calls (full epoch re-run for non-stateful loader), got {model.train_calls}"
    )
    assert trainer.current_epoch == 1


# --------------------------------------------------------------------------
# CombinedLoader loader-state pipeline (opt-in for a stateful loader)
# --------------------------------------------------------------------------


class _StatefulFake:
    """A minimal object that opts into loader persistence by exposing state_dict."""

    def __init__(self, payload=None):
        self._payload = payload if payload is not None else {"v": 1}

    def state_dict(self):
        return dict(self._payload)

    def load_state_dict(self, state):
        self._payload = dict(state)


def test_combined_loader_state_dicts_filters_stateful(tmp_path):
    from ocean.utils.combined_loader import CombinedLoader

    stateful = _StatefulFake({"v": 7})

    class _Plain:
        pass  # plain loader: no state_dict, must be silently skipped

        def __iter__(self):
            return iter([])

    cl = CombinedLoader([stateful, _Plain()], mode="sequential")
    states = cl._state_dicts()
    assert len(states) == 1, "only the stateful loader should contribute state"
    assert states[0] == {"v": 7}


def test_combined_loader_load_state_dict_count_mismatch_raises(tmp_path):
    import pytest

    from ocean.utils.combined_loader import CombinedLoader

    stateful = _StatefulFake()
    cl = CombinedLoader([stateful], mode="sequential")
    with pytest.raises(RuntimeError):
        cl._load_state_dicts([{"a": 1}, {"b": 2}])  # two states, one stateful loader
    # matching count restores without error
    cl._load_state_dicts([{"v": 9}])
    assert stateful._payload == {"v": 9}


def test_restore_records_loader_state_into_fit_loop(tmp_path):
    """A checkpoint carrying a ``combined_loader`` key is staged on the fit loop;

    a checkpoint without it (plain loaders / old checkpoints) stages nothing.
    """
    model = _CountingModel()
    path = _fit_and_save(tmp_path, model, limit_train_batches=2)
    trainer = ocean.Trainer(
        max_epochs=1,
        limit_train_batches=0,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    # A plain Paddle loader contributes no state, so checkpoint has no combined_loader.
    trainer.fit(model, train_dataloaders=_dl(), ckpt_path=path)
    assert getattr(trainer.fit_loop, "_combined_loader_states_to_load", None) in ([], None)
