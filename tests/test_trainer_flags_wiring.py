"""Tests that Trainer flags which were stored and never read now act.

Covers:
- min_steps holds off an early stop
- overfit_batches pins the batch order, so the same batches really repeat
- sync_batchnorm converts the batch-norm layers
- plugins= installs a precision plugin
- enable_model_summary controls whether the summary appears
"""

import contextlib
import io

import paddle
import pytest

import ocean
from ocean.callbacks.model_summary import ModelSummary
from ocean.plugins.layer_sync import LayerSync
from ocean.plugins.precision.amp import MixedPrecision


class CountingModel(ocean.Model):
    def __init__(self, stop_immediately=False):
        super().__init__()
        self.bn = paddle.nn.BatchNorm1D(4)
        self.linear = paddle.nn.Linear(4, 2)
        self.batches = []
        self.stop_immediately = stop_immediately

    def forward(self, x):
        return self.linear(self.bn(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.batches.append(round(float(x.sum()), 3))
        if self.stop_immediately and self._trainer.current_epoch == 0 and batch_idx == 0:
            self._trainer.should_stop = True
        return paddle.nn.functional.cross_entropy(self(x), y)

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.1, parameters=self.parameters())


def make_loader(n=40, bs=8, shuffle=False):
    ds = paddle.io.TensorDataset([paddle.randn([n, 4]), paddle.randint(0, 2, [n])])
    return paddle.io.DataLoader(ds, batch_size=bs, shuffle=shuffle)


def make_trainer(**kwargs):
    for key, value in dict(
        verbose=0, logger=False, enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False
    ).items():
        kwargs.setdefault(key, value)
    return ocean.Trainer(**kwargs)


# ── min_steps ────────────────────────────────────────────────────────────────


def test_min_steps_holds_off_an_early_stop():
    model = CountingModel(stop_immediately=True)
    make_trainer(max_epochs=9, min_steps=20).fit(model, train_dataloaders=make_loader())
    assert len(model.batches) >= 20


def test_without_min_steps_an_early_stop_is_immediate():
    model = CountingModel(stop_immediately=True)
    make_trainer(max_epochs=9).fit(model, train_dataloaders=make_loader())
    assert len(model.batches) == 5  # one epoch, then stop


def test_min_epochs_still_holds():
    model = CountingModel(stop_immediately=True)
    make_trainer(max_epochs=9, min_epochs=3).fit(model, train_dataloaders=make_loader())
    assert len(model.batches) == 15  # three epochs of five batches


# ── overfit_batches ──────────────────────────────────────────────────────────


def test_overfit_batches_repeats_the_same_batches():
    """The batch limit alone gives a different subset every epoch when the
    loader shuffles, which overfits nothing."""
    model = CountingModel()
    make_trainer(max_epochs=3, overfit_batches=2).fit(model, train_dataloaders=make_loader(shuffle=True))

    assert len(model.batches) == 6
    assert model.batches[0:2] == model.batches[2:4] == model.batches[4:6]


def test_without_overfit_shuffling_is_left_alone():
    loader = make_loader(shuffle=True)
    make_trainer(max_epochs=1).fit(CountingModel(), train_dataloaders=loader)
    assert isinstance(loader.batch_sampler.sampler, paddle.io.RandomSampler)


# ── sync_batchnorm ───────────────────────────────────────────────────────────


def test_sync_batchnorm_converts_the_layers(monkeypatch):
    """Applied only when there is more than one process to synchronise with."""
    calls = []

    class SpySync(LayerSync):
        def sync(self, model):
            calls.append(type(model).__name__)

    trainer = make_trainer(max_epochs=1, sync_batchnorm=True, plugins=[SpySync()])
    monkeypatch.setattr(type(trainer.strategy), "world_size", property(lambda self: 2))
    trainer.fit(CountingModel(), train_dataloaders=make_loader())

    assert calls == ["CountingModel"]


def test_sync_batchnorm_is_a_no_op_on_one_process(capsys):
    """SyncBatchNorm across a single device means nothing, and on some CPU
    builds its kernel is not registered at all — converting would trade a
    do-nothing flag for a crash."""
    model = CountingModel()
    make_trainer(max_epochs=1, sync_batchnorm=True).fit(model, train_dataloaders=make_loader())

    assert isinstance(model.bn, paddle.nn.BatchNorm1D)
    assert not isinstance(model.bn, paddle.nn.SyncBatchNorm)
    assert "no effect with a single process" in capsys.readouterr().out


def test_without_the_flag_batch_norm_is_untouched():
    model = CountingModel()
    make_trainer(max_epochs=1).fit(model, train_dataloaders=make_loader())
    assert isinstance(model.bn, paddle.nn.BatchNorm1D)
    assert not isinstance(model.bn, paddle.nn.SyncBatchNorm)


# ── plugins ──────────────────────────────────────────────────────────────────


def test_precision_plugin_from_plugins_is_installed():
    plugin = MixedPrecision("16-mixed")
    trainer = make_trainer(plugins=[plugin])
    assert trainer.strategy.precision_plugin is plugin


def test_precision_plugin_overrides_the_precision_string():
    trainer = make_trainer(precision="32", plugins=[MixedPrecision("16-mixed")])
    assert trainer.strategy.precision_plugin.precision == "16-mixed"


def test_layer_sync_plugin_is_installed():
    class SpySync(LayerSync):
        def sync(self, model):
            pass

    plugin = SpySync()
    assert make_trainer(plugins=[plugin])._layer_sync is plugin


def test_unknown_plugin_is_reported(capsys):
    make_trainer(plugins=[object()])
    assert "Ignoring unsupported plugin" in capsys.readouterr().out


# ── enable_model_summary ─────────────────────────────────────────────────────


def test_model_summary_is_added_when_enabled():
    trainer = ocean.Trainer(verbose=0, logger=False, enable_checkpointing=False, enable_progress_bar=False)
    assert any(isinstance(cb, ModelSummary) for cb in trainer.callbacks)


def test_model_summary_is_absent_when_disabled():
    trainer = make_trainer()
    assert not any(isinstance(cb, ModelSummary) for cb in trainer.callbacks)


def test_model_summary_prints():
    buffer = io.StringIO()
    trainer = ocean.Trainer(
        max_epochs=1, verbose=0, logger=False, enable_checkpointing=False, enable_progress_bar=False
    )
    with contextlib.redirect_stdout(buffer):
        trainer.fit(CountingModel(), train_dataloaders=make_loader())
    assert "Params" in buffer.getvalue() or "Name" in buffer.getvalue()


def test_a_user_supplied_summary_is_not_duplicated():
    trainer = ocean.Trainer(
        verbose=0, logger=False, enable_checkpointing=False, enable_progress_bar=False, callbacks=[ModelSummary()]
    )
    assert sum(isinstance(cb, ModelSummary) for cb in trainer.callbacks) == 1


@pytest.mark.parametrize("flag", [True, False])
def test_summary_flag_round_trips(flag):
    trainer = make_trainer(enable_model_summary=flag)
    assert any(isinstance(cb, ModelSummary) for cb in trainer.callbacks) is flag
