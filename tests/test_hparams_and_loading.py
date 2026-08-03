"""Tests for the hyperparameter system and ``load_from_checkpoint``.

``HyperparametersMixin`` used to be exported but inherited by nothing, so
``save_hyperparameters()`` did not exist on ``Model``/``DataModule`` and
``load_from_checkpoint`` could never rebuild a model from its stored
hyperparameters.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paddle
import pytest

import ocean
from ocean.core.mixins import AttributeDict, HyperparametersMixin


class HParamModel(ocean.Model):
    def __init__(self, hidden=8, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.linear = paddle.nn.Linear(4, hidden)

    def forward(self, x):
        return self.linear(x)


class HParamDataModule(ocean.DataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.save_hyperparameters()


# --- the mixin is actually inherited ---------------------------------------


def test_model_inherits_hyperparameters_mixin():
    assert issubclass(ocean.Model, HyperparametersMixin)
    assert issubclass(ocean.DataModule, HyperparametersMixin)


def test_paddle_layer_machinery_survives_the_mixin():
    """The mixin sits ahead of nn.Layer, so Layer.__init__ must still run."""
    model = HParamModel(hidden=6)
    assert len(model.parameters()) == 2
    assert len(model.sublayers()) == 1
    assert model(paddle.randn([2, 4])).shape == [2, 6]


def test_save_hyperparameters_captures_init_args():
    model = HParamModel(hidden=7, lr=0.05)
    assert dict(model.hparams) == {"hidden": 7, "lr": 0.05}
    assert model.hparams_initial == {"hidden": 7, "lr": 0.05}
    assert model.hparams.hidden == 7  # attribute-style access


def test_datamodule_save_hyperparameters():
    dm = HParamDataModule(batch_size=64)
    assert dict(dm.hparams) == {"batch_size": 64}


def test_hparams_default_empty_and_assignable():
    model = ocean.Model()
    assert dict(model.hparams) == {}
    assert model.hparams_initial == {}
    model.hparams = {"a": 1}
    assert isinstance(model.hparams, AttributeDict)
    assert model.hparams["a"] == 1


# --- load_from_checkpoint ---------------------------------------------------


def test_load_from_checkpoint_rebuilds_from_hparams(tmp_path):
    """The saved hyperparameters must drive re-construction of the model."""
    model = HParamModel(hidden=7, lr=0.05)
    path = str(tmp_path / "ckpt.pdparams")
    model.save_checkpoint(path)

    loaded = HParamModel.load_from_checkpoint(path)

    assert loaded.linear.weight.shape[1] == 7, "hparams were ignored; got default width"
    assert loaded.hparams_initial == {"hidden": 7, "lr": 0.05}
    assert bool((loaded.linear.weight == model.linear.weight).all())


def test_load_from_checkpoint_kwargs_override_stored_hparams(tmp_path):
    model = HParamModel(hidden=7, lr=0.05)
    path = str(tmp_path / "ckpt.pdparams")
    model.save_checkpoint(path)

    loaded = HParamModel.load_from_checkpoint(path, lr=0.9)
    assert loaded.hparams["lr"] == 0.9
    assert loaded.hparams["hidden"] == 7


def test_load_from_checkpoint_reads_legacy_hparams_key(tmp_path):
    path = str(tmp_path / "legacy.pdparams")
    paddle.save({"state_dict": HParamModel(hidden=6).state_dict(), "hparams": {"hidden": 6}}, path)

    assert HParamModel.load_from_checkpoint(path).linear.weight.shape[1] == 6


def test_load_from_checkpoint_ignores_hparams_init_cannot_accept(tmp_path):
    """A hyperparameter removed from __init__ must not blow up old checkpoints."""
    path = str(tmp_path / "stale.pdparams")
    paddle.save(
        {"state_dict": HParamModel(hidden=6).state_dict(), "hyper_parameters": {"hidden": 6, "removed_arg": 1}},
        path,
    )

    assert HParamModel.load_from_checkpoint(path).linear.weight.shape[1] == 6


def test_datamodule_load_from_checkpoint(tmp_path):
    path = str(tmp_path / "dm.pdparams")
    paddle.save({"hyper_parameters": {"batch_size": 128}}, path)

    assert HParamDataModule.load_from_checkpoint(path).hparams["batch_size"] == 128


# --- strict ----------------------------------------------------------------


class TwoLayerModel(ocean.Model):
    def __init__(self, hidden=8):
        super().__init__()
        self.save_hyperparameters()
        self.linear = paddle.nn.Linear(4, hidden)
        self.head = paddle.nn.Linear(hidden, 2)

    def forward(self, x):
        return self.head(self.linear(x))


def _partial_checkpoint(tmp_path):
    model = TwoLayerModel()
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("head")}
    path = str(tmp_path / "partial.pdparams")
    paddle.save({"state_dict": state, "hyper_parameters": {"hidden": 8}}, path)
    return path


def test_load_from_checkpoint_strict_rejects_missing_keys(tmp_path):
    """`strict` was a no-op: `set_dict` IS `set_state_dict` in Paddle."""
    with pytest.raises(RuntimeError, match="Missing key"):
        TwoLayerModel.load_from_checkpoint(_partial_checkpoint(tmp_path))


def test_load_from_checkpoint_non_strict_allows_missing_keys(tmp_path):
    model = TwoLayerModel.load_from_checkpoint(_partial_checkpoint(tmp_path), strict=False)
    assert model.head.weight is not None


def test_load_checkpoint_strict_rejects_missing_keys(tmp_path):
    """The instance-level loader had the same fake-strict bug."""
    path = _partial_checkpoint(tmp_path)
    with pytest.raises(RuntimeError, match="Missing key"):
        TwoLayerModel().load_checkpoint(path)

    TwoLayerModel().load_checkpoint(path, strict=False)  # does not raise


# --- checkpoints written by the trainer ------------------------------------


class TrainableHParamModel(HParamModel):
    def training_step(self, batch, batch_idx):
        x, y = batch
        return paddle.nn.functional.cross_entropy(self.linear(x)[:, :2], y)

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())


def test_trainer_checkpoint_roundtrips_hparams(tmp_path):
    ds = paddle.io.TensorDataset([paddle.randn([16, 4]), paddle.randint(0, 2, [16])])
    loader = paddle.io.DataLoader(ds, batch_size=4)

    ckpt = ocean.ModelCheckpoint(dirpath=str(tmp_path), save_top_k=1)
    trainer = ocean.Trainer(max_epochs=1, callbacks=[ckpt], verbose=0, enable_progress_bar=False, logger=False)
    trainer.fit(TrainableHParamModel(hidden=5), train_dataloaders=loader)

    saved = paddle.load(ckpt.best_model_path)
    assert saved["hyper_parameters"]["hidden"] == 5
    assert type(saved["hyper_parameters"]) is dict  # paddle.save rejects dict subclasses
