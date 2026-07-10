"""T4: reload_dataloaders_every_n_epochs actually rebuilds the train dataloader.

Previously this parameter was accepted and validated but never acted upon.
It now re-runs the datamodule's setup('fit') + train_dataloader() every N epochs.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paddle

import ocean


def _make_loader(num_samples):
    ds = paddle.io.TensorDataset([
        paddle.randn([num_samples, 4]),
        paddle.randint(0, 2, [num_samples]),
    ])
    return paddle.io.DataLoader(ds, batch_size=4)


class _GrowingDataModule(ocean.DataModule):
    """Yields more batches each time train_dataloader() is called."""

    def __init__(self):
        super().__init__()
        self.setup_calls = 0
        self.dl_calls = 0

    def setup(self, stage=None):
        self.setup_calls += 1

    def train_dataloader(self):
        self.dl_calls += 1
        # 1st call: 8 samples (2 batches); later calls: 16 samples (4 batches)
        n = 8 if self.dl_calls == 1 else 16
        return _make_loader(n)

    def val_dataloader(self):
        return _make_loader(8)


class _CountingModel(ocean.Model):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(4, 2)
        self.per_epoch_batches = []
        self._cur = 0

    def forward(self, x):
        return self.linear(x)

    def on_train_epoch_start(self):
        self._cur = 0

    def training_step(self, batch, batch_idx):
        self._cur += 1
        return paddle.nn.functional.cross_entropy(self(batch[0]), batch[1])

    def on_train_epoch_end(self):
        self.per_epoch_batches.append(self._cur)

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())


def test_reload_every_epoch_rebuilds_loader():
    model = _CountingModel()
    dm = _GrowingDataModule()
    trainer = ocean.Trainer(
        max_epochs=3,
        reload_dataloaders_every_n_epochs=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=dm)
    # epoch 0 uses the initial loader (2 batches); epochs 1 and 2 reload (4 batches)
    assert model.per_epoch_batches == [2, 4, 4], model.per_epoch_batches


def test_no_reload_when_zero():
    model = _CountingModel()
    dm = _GrowingDataModule()
    trainer = ocean.Trainer(
        max_epochs=3,
        reload_dataloaders_every_n_epochs=0,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=dm)
    # never reloaded: every epoch uses the initial 2-batch loader
    assert model.per_epoch_batches == [2, 2, 2], model.per_epoch_batches


def test_reload_every_two_epochs():
    model = _CountingModel()
    dm = _GrowingDataModule()
    trainer = ocean.Trainer(
        max_epochs=4,
        reload_dataloaders_every_n_epochs=2,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=dm)
    # reload only at epoch 2: epochs 0,1 = 2 batches; epochs 2,3 = 4 batches
    assert model.per_epoch_batches == [2, 2, 4, 4], model.per_epoch_batches
