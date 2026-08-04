"""Tests for the setup/teardown lifecycle hooks.

Covers:
- setup and teardown fire on the datamodule, the callbacks and the model
- they fire for every entry point, not just fit
- they receive the stage that is actually running
- setup runs before the datasets are read, and only once
"""

import paddle
import pytest

import ocean
from ocean.callbacks.callback import Callback
from ocean.datamodule import DataModule
from ocean.model import Model

# ── Recording fixtures ───────────────────────────────────────────────────────


class Recorder:
    def __init__(self):
        self.events = []


class SpyCallback(Callback):
    def __init__(self, recorder):
        self.recorder = recorder

    def setup(self, trainer, model, stage):
        self.recorder.events.append(("callback.setup", stage))

    def teardown(self, trainer, model, stage):
        self.recorder.events.append(("callback.teardown", stage))


class SpyModel(Model):
    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder
        self.linear = paddle.nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)

    def setup(self, stage):
        self.recorder.events.append(("model.setup", stage))

    def teardown(self, stage):
        self.recorder.events.append(("model.teardown", stage))

    def training_step(self, batch, batch_idx):
        x, y = batch
        return paddle.nn.functional.cross_entropy(self(x), y)

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch):
        return self(batch[0])

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())


class SpyDataModule(DataModule):
    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder
        self.dataset = None

    def setup(self, stage):
        self.recorder.events.append(("datamodule.setup", stage))
        self.dataset = paddle.io.TensorDataset([paddle.randn([16, 4]), paddle.randint(0, 2, [16])])

    def teardown(self, stage):
        self.recorder.events.append(("datamodule.teardown", stage))

    def _loader(self):
        # Deliberately depends on setup() having run: reading the datasets
        # before setup is what the ordering guarantee protects against.
        return paddle.io.DataLoader(self.dataset, batch_size=8)

    train_dataloader = val_dataloader = test_dataloader = predict_dataloader = _loader


def make_trainer(recorder):
    return ocean.Trainer(
        max_epochs=1,
        verbose=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        callbacks=[SpyCallback(recorder)],
    )


def run(entry_point):
    recorder = Recorder()
    getattr(make_trainer(recorder), entry_point)(SpyModel(recorder), datamodule=SpyDataModule(recorder))
    return recorder.events


# ── Every entry point, correct stage ─────────────────────────────────────────


@pytest.mark.parametrize("entry_point", ["fit", "validate", "test", "predict"])
def test_setup_and_teardown_fire_with_the_running_stage(entry_point):
    assert run(entry_point) == [
        ("datamodule.setup", entry_point),
        ("callback.setup", entry_point),
        ("model.setup", entry_point),
        ("datamodule.teardown", entry_point),
        ("callback.teardown", entry_point),
        ("model.teardown", entry_point),
    ]


@pytest.mark.parametrize("entry_point", ["fit", "validate", "test", "predict"])
def test_setup_runs_once(entry_point):
    events = run(entry_point)
    assert [e for e in events if e[0] == "datamodule.setup"] == [("datamodule.setup", entry_point)]


def test_setup_precedes_the_dataloader_request():
    """The datamodule builds its datasets in setup, so a loader requested first
    would see None."""
    recorder = Recorder()
    trainer = make_trainer(recorder)
    datamodule = SpyDataModule(recorder)
    trainer.fit(SpyModel(recorder), datamodule=datamodule)
    assert datamodule.dataset is not None


# ── Without a datamodule ─────────────────────────────────────────────────────


def test_hooks_fire_without_a_datamodule():
    recorder = Recorder()
    loader = paddle.io.DataLoader(
        paddle.io.TensorDataset([paddle.randn([16, 4]), paddle.randint(0, 2, [16])]), batch_size=8
    )
    make_trainer(recorder).fit(SpyModel(recorder), train_dataloaders=loader)

    assert ("callback.setup", "fit") in recorder.events
    assert ("model.setup", "fit") in recorder.events
    assert ("callback.teardown", "fit") in recorder.events
    assert ("model.teardown", "fit") in recorder.events


def test_teardown_runs_even_when_the_run_raises():
    class Boom(SpyModel):
        def training_step(self, batch, batch_idx):
            raise RuntimeError("boom")

    recorder = Recorder()
    loader = paddle.io.DataLoader(
        paddle.io.TensorDataset([paddle.randn([16, 4]), paddle.randint(0, 2, [16])]), batch_size=8
    )
    with pytest.raises(RuntimeError, match="boom"):
        make_trainer(recorder).fit(Boom(recorder), train_dataloaders=loader)

    assert ("model.teardown", "fit") in recorder.events
