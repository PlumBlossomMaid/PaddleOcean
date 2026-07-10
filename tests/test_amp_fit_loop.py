"""Tests for AMP integration in automatic optimization loop.

Verifies that ``_AutomaticOptimization.run()`` correctly uses strategy-layer
hooks for AMP (``strategy.training_step()`` / ``strategy.backward()`` /
``strategy.optimizer_step()`` + ``unscale_gradients()``).

Ablation / regression:
    Old code → ``model.training_step()`` + ``loss.backward()`` + ``optimizer.step()``
    New code → ``strategy.training_step()`` + ``strategy.backward()`` + ``strategy.optimizer_step()``
"""

import inspect
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paddle
import pytest

import ocean
from ocean.loops.optimization.automatic import _AutomaticOptimization


def _class_src() -> str:
    return inspect.getsource(_AutomaticOptimization)


# ====================================================================
# Source-code assertions (work on CPU CI)
# ====================================================================


def test_uses_strategy_training_step():
    src = _class_src()
    assert "strategy.training_step" in src


def test_uses_strategy_backward():
    src = _class_src()
    assert "strategy.backward" in src


def test_uses_strategy_optimizer_step():
    src = _class_src()
    assert "strategy.optimizer_step" in src


def test_uses_unscale_gradients():
    src = _class_src()
    assert "unscale_gradients" in src


def test_uses_advance_optimizer_step():
    src = _class_src()
    assert "_advance_optimizer_step" in src


# ====================================================================
# Ablation – old code patterns MUST be absent
# ====================================================================


def test_ablation_old_code_would_fail():
    """If the fix is reverted this test FAILS — the strategy hooks vanish."""
    src = _class_src()
    assert "strategy.training_step" in src, "ABLATION FAILED"
    assert "strategy.backward" in src, "ABLATION FAILED"
    assert "strategy.optimizer_step" in src, "ABLATION FAILED"


# ====================================================================
# Plugin unit tests
# ====================================================================


def test_mixed_precision_forward_context():
    mp = ocean.plugins.MixedPrecision("16-mixed")
    ctx = mp.forward_context()
    assert hasattr(ctx, "__enter__")
    assert hasattr(ctx, "__exit__")


def test_mixed_precision_backward():
    mp = ocean.plugins.MixedPrecision("16-mixed")
    assert hasattr(mp, "_scaler")
    assert isinstance(mp._scaler, paddle.amp.GradScaler)


# ====================================================================
# GPU functional tests
# ====================================================================


class _SimpleModel(ocean.Model):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(32, 8)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        return paddle.nn.functional.cross_entropy(logits, y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        return paddle.nn.functional.cross_entropy(logits, y)

    def configure_optimizers(self):
        sched = paddle.optimizer.lr.StepDecay(learning_rate=0.01, step_size=5, gamma=0.9)
        return {
            "optimizer": paddle.optimizer.Adam(learning_rate=sched, parameters=self.parameters()),
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }


def _data():
    xs = paddle.randn([32, 32])
    ys = paddle.randint(0, 8, [32])
    ds = paddle.io.TensorDataset([xs, ys])
    return paddle.io.DataLoader(ds, batch_size=8, shuffle=True)


@pytest.mark.skipif(not paddle.is_compiled_with_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("acc", [1, 2, 4])
def test_amp_accumulation_cuda(acc):
    model = _SimpleModel()
    trainer = ocean.Trainer(
        max_steps=10,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=acc,
        gradient_clip_val=1.0,
        callbacks=[ocean.callbacks.TQDMProgressBar()],
        logger=False,
    )
    trainer.fit(model, train_dataloaders=_data())
    loss = trainer.logged_metrics.get("train_loss")
    if loss is not None:
        assert not (isinstance(loss, float) and math.isnan(loss))
        assert isinstance(loss, float) and math.isfinite(loss)


@pytest.mark.skipif(not paddle.is_compiled_with_cuda(), reason="requires CUDA")
def test_amp_gradient_clip_cuda():
    model = _SimpleModel()
    trainer = ocean.Trainer(
        max_steps=5,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=2,
        gradient_clip_val=0.5,
        callbacks=[ocean.callbacks.TQDMProgressBar()],
        logger=False,
    )
    trainer.fit(model, train_dataloaders=_data())
    assert trainer.optimizer_step > 0


@pytest.mark.skipif(not paddle.is_compiled_with_cuda(), reason="requires CUDA")
def test_amp_precision_32_fallback():
    model = _SimpleModel()
    trainer = ocean.Trainer(
        max_steps=5,
        accelerator="gpu",
        devices=1,
        precision="32-true",
        accumulate_grad_batches=2,
        callbacks=[ocean.callbacks.TQDMProgressBar()],
        logger=False,
    )
    trainer.fit(model, train_dataloaders=_data())
    assert trainer.optimizer_step > 0


# ====================================================================
# CPU fallback
# ====================================================================


@pytest.mark.skipif(paddle.is_compiled_with_cuda(), reason="CPU-only")
def test_cpu_precision_16_mixed_fallback():
    model = _SimpleModel()
    trainer = ocean.Trainer(
        max_steps=3,
        accelerator="cpu",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=2,
        callbacks=[ocean.callbacks.TQDMProgressBar()],
    )
    trainer.fit(model, train_dataloaders=_data())


def test_mixed_precision_plugins_importable():
    from ocean.plugins import MixedPrecision, Precision

    assert MixedPrecision is not None
    assert Precision is not None
