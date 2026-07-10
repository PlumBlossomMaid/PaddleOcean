"""CB1/CB2: model callbacks override trainer callbacks; checkpoints run last.

CB1 — a callback returned by Model.configure_callbacks() replaces a trainer
callback of the same (or super) type instead of coexisting with it.
CB2 — checkpoint callbacks are ordered after all others, so a monitored metric
is up to date when a checkpoint is written.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paddle

import ocean
from ocean.callbacks.callback import Callback
from ocean.callbacks.checkpoint import ModelCheckpoint


class _MetricCallback(Callback):
    pass


class _CustomCheckpoint(ModelCheckpoint):
    pass


class _Model(ocean.Model):
    def __init__(self, model_callbacks=None):
        super().__init__()
        self.linear = paddle.nn.Linear(4, 2)
        self._model_callbacks = model_callbacks or []

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        return paddle.nn.functional.cross_entropy(self(batch[0]), batch[1])

    def configure_callbacks(self):
        return self._model_callbacks

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())


def _attach(trainer, model):
    model._trainer = trainer
    trainer._model = model
    trainer._callback_connector._attach_model_callbacks()


# ── CB2: reorder ────────────────────────────────────────────────────────────


def test_default_checkpoint_reordered_last():
    trainer = ocean.Trainer(
        max_epochs=1,
        callbacks=[_MetricCallback()],
        enable_checkpointing=True,
        enable_progress_bar=False,
        logger=False,
    )
    assert isinstance(trainer.callbacks[-1], ModelCheckpoint)
    assert isinstance(trainer.callbacks[0], _MetricCallback)


def test_checkpoint_last_even_with_progress_bar():
    trainer = ocean.Trainer(
        max_epochs=1,
        callbacks=[_MetricCallback()],
        enable_checkpointing=True,
        enable_progress_bar=True,
        logger=False,
    )
    assert isinstance(trainer.callbacks[-1], ModelCheckpoint)


# ── CB1: dedup / override ────────────────────────────────────────────────────


def test_model_checkpoint_overrides_default():
    trainer = ocean.Trainer(
        max_epochs=1,
        enable_checkpointing=True,
        enable_progress_bar=False,
        logger=False,
    )
    # default ModelCheckpoint present
    assert sum(type(c) is ModelCheckpoint for c in trainer.callbacks) == 1

    _attach(trainer, _Model(model_callbacks=[_CustomCheckpoint(dirpath=".")]))

    # default ModelCheckpoint replaced by the model's subclass; no duplicate
    assert sum(type(c) is ModelCheckpoint for c in trainer.callbacks) == 0
    assert sum(type(c) is _CustomCheckpoint for c in trainer.callbacks) == 1
    # still ordered last (checkpoint group)
    assert isinstance(trainer.callbacks[-1], _CustomCheckpoint)


def test_model_callback_of_new_type_is_appended():
    trainer = ocean.Trainer(
        max_epochs=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    _attach(trainer, _Model(model_callbacks=[_MetricCallback()]))
    assert sum(type(c) is _MetricCallback for c in trainer.callbacks) == 1


def test_no_model_callbacks_leaves_list_unchanged():
    trainer = ocean.Trainer(
        max_epochs=1,
        callbacks=[_MetricCallback()],
        enable_checkpointing=True,
        enable_progress_bar=False,
        logger=False,
    )
    before = list(trainer.callbacks)
    _attach(trainer, _Model(model_callbacks=[]))
    assert trainer.callbacks == before


def test_reorder_callbacks_helper_direct():
    from ocean.callbacks.batch_size_finder import BatchSizeFinder
    from ocean.trainer.connectors import _CallbackConnector

    ckpt = ModelCheckpoint(dirpath=".")
    metric = _MetricCallback()
    tuner = BatchSizeFinder()
    ordered = _CallbackConnector._reorder_callbacks([ckpt, metric, tuner])
    assert isinstance(ordered[0], BatchSizeFinder)  # tuner first
    assert isinstance(ordered[-1], ModelCheckpoint)  # checkpoint last
    assert isinstance(ordered[1], _MetricCallback)  # other in middle
