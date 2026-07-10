"""C1 regression: model-defined callbacks get their checkpoint state restored.

A callback returned from ``Model.configure_callbacks()`` must be attached before
the checkpoint is restored, otherwise its saved ``state_dict`` is silently
dropped on resume (it is not yet in ``trainer.callbacks`` at restore time).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paddle

import ocean
from ocean.callbacks.callback import Callback


class _StatefulCallback(Callback):
    """A callback with resumable state, provided via configure_callbacks()."""

    def __init__(self):
        self.counter = 0

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        self.counter += 1

    def state_dict(self):
        return {"counter": self.counter}

    def load_state_dict(self, state_dict):
        self.counter = state_dict.get("counter", 0)


class _ModelWithCallback(ocean.Model):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(10, 2)
        self._cb = _StatefulCallback()

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = paddle.nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_callbacks(self):
        return [self._cb]

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())


def _dl(num_samples=32, batch_size=8):
    ds = paddle.io.TensorDataset([
        paddle.randn([num_samples, 10]),
        paddle.randint(0, 2, [num_samples]),
    ])
    return paddle.io.DataLoader(ds, batch_size=batch_size)


def test_model_callback_state_restored_on_resume(tmp_path):
    # First run: train and save a checkpoint carrying the callback's counter.
    model = _ModelWithCallback()
    trainer = ocean.Trainer(
        max_epochs=1,
        limit_train_batches=4,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=_dl())
    saved_counter = model._cb.counter
    assert saved_counter == 4  # 4 training batches

    ckpt = os.path.join(str(tmp_path), "resume.pdparams")
    trainer.save_checkpoint(ckpt)

    # Second run: resume. The fresh model's callback starts at 0 but must be
    # restored to saved_counter because attach happens before restore.
    model2 = _ModelWithCallback()
    assert model2._cb.counter == 0
    trainer2 = ocean.Trainer(
        max_epochs=1,
        limit_train_batches=0,  # don't train further; just exercise restore
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer2.fit(model2, train_dataloaders=_dl(), ckpt_path=ckpt)

    assert model2._cb.counter == saved_counter, (
        f"model-defined callback state not restored: {model2._cb.counter} != {saved_counter}"
    )
