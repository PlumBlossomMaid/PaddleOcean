"""Tests for ModelSummary In/Out size capture.

The In size / Out size columns used to always print ``-``. They are now filled
from a single example forward pass driven by ``model.example_input_array``,
captured with forward hooks, and left as ``-`` when no example is available or
the forward pass fails.
"""

from __future__ import annotations

import paddle
import paddle.nn as nn

import ocean
from ocean.callbacks.model_summary import ModelSummary, _parse_shape


class _Model(ocean.Model):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 2)
        self.example_input_array = paddle.randn([3, 4])

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self(x).mean()


def test_parse_shape():
    assert _parse_shape(paddle.randn([3, 4])) == [3, 4]
    assert _parse_shape("not a tensor") == "-"
    assert _parse_shape((paddle.randn([2, 2]),)) == [2, 2]
    assert _parse_shape([paddle.randn([2, 2]), paddle.randn([5])]) == [[2, 2], [5]]


def test_summary_captures_io_sizes():
    summary = ModelSummary(max_depth=1)._get_summary(_Model())
    # The Linear child sees [3, 4] in and produces [3, 2] out.
    assert "[3, 4]" in summary
    assert "[3, 2]" in summary


def test_summary_without_example_leaves_sizes_unknown():
    model = _Model()
    model.example_input_array = None
    summary = ModelSummary(max_depth=1)._get_summary(model)
    # In/Out columns fall back to the unknown marker.
    assert "| - | - |" in summary
    # Params are still reported.
    assert "Total params:" in summary


def test_summary_survives_forward_failure():
    class _NoForward(ocean.Model):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Linear(4, 2)
            self.example_input_array = paddle.randn([3, 4])

        # forward raises -> sizes must stay unknown, no crash
        def forward(self, x):
            raise RuntimeError("no forward here")

        def configure_optimizers(self):
            return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())

        def training_step(self, batch, batch_idx):
            return paddle.to_tensor(0.0)

    summary = ModelSummary(max_depth=1)._get_summary(_NoForward())
    assert "| - | - |" in summary


def test_summary_in_real_fit_does_not_crash(tmp_path):
    model = _Model()
    trainer = ocean.Trainer(
        default_root_dir=str(tmp_path),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        callbacks=[ModelSummary(max_depth=1)],
    )
    loader = paddle.io.DataLoader(paddle.io.TensorDataset([paddle.randn([10, 4])]), batch_size=5)
    trainer.fit(model, loader)
    # Model is left in training mode after the summary's eval() forward pass.
    assert model.training
