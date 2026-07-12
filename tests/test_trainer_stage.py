"""T1: the running stage is reflected on the Trainer during each loop.

trainer.training / validating / testing / predicting (and sanity_checking) must
be correct inside the corresponding step, since user code and metric fx-keying
rely on them. Previously the stage was never set, so they were always False.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paddle

import ocean


class _StageModel(ocean.Model):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(4, 2)
        self.train_flags = []
        self.val_flags = []
        self.test_flags = []

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        t = self._trainer
        self.train_flags.append((t.training, t.validating, t.testing))
        return paddle.nn.functional.cross_entropy(self(batch[0]), batch[1])

    def validation_step(self, batch, batch_idx):
        t = self._trainer
        self.val_flags.append((t.training, t.validating, t.testing))
        self.log("vl", paddle.nn.functional.cross_entropy(self(batch[0]), batch[1]), on_epoch=True)

    def test_step(self, batch, batch_idx):
        t = self._trainer
        self.test_flags.append((t.training, t.validating, t.testing))
        self.log("tl", paddle.nn.functional.cross_entropy(self(batch[0]), batch[1]), on_epoch=True)

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())


def _dl(n=8):
    ds = paddle.io.TensorDataset([paddle.randn([n, 4]), paddle.randint(0, 2, [n])])
    return paddle.io.DataLoader(ds, batch_size=4)


def test_training_stage_flag():
    model = _StageModel()
    trainer = ocean.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_dl())
    assert model.train_flags, "training_step never ran"
    assert all(f == (True, False, False) for f in model.train_flags)


def test_midepoch_validation_stage_flag():
    model = _StageModel()
    trainer = ocean.Trainer(
        max_epochs=1,
        val_check_interval=2,
        limit_train_batches=2,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_dl(8), val_dataloaders=_dl(4))
    assert model.val_flags, "validation_step never ran"
    assert all(f == (False, True, False) for f in model.val_flags)
    # training flags stay correct after the val interlude
    assert all(f == (True, False, False) for f in model.train_flags)


def test_standalone_validate_stage_flag():
    model = _StageModel()
    trainer = ocean.Trainer(num_sanity_val_steps=0, logger=False, enable_checkpointing=False)
    trainer.validate(model, dataloaders=_dl(4))
    assert model.val_flags
    assert all(f == (False, True, False) for f in model.val_flags)


def test_standalone_test_stage_flag():
    model = _StageModel()
    trainer = ocean.Trainer(logger=False, enable_checkpointing=False)
    trainer.test(model, dataloaders=_dl(4))
    assert model.test_flags
    assert all(f == (False, False, True) for f in model.test_flags)


def test_stage_cleared_after_fit():
    model = _StageModel()
    trainer = ocean.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_dl())
    # During the run the training stage was active...
    assert model.train_flags[-1] == (True, False, False)
    # ...and it is cleared once fit finishes (teardown resets state.stage).
    assert trainer.training is False
    assert trainer.validating is False
    assert trainer.state.stage is None
