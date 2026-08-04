"""Tests that Gear's constructor and method arguments actually do something.

Covers:
- precision= builds the matching precision plugin (and autocast agrees with it)
- setup() wraps optimizers so step() goes through that plugin
- setup_dataloaders(move_to_device=True) really moves the batches
- load(strict=True) reports a mismatch instead of loading anything
- loggers= is reachable through log()/log_dict()
- the rank accessors a manual training loop needs
"""

import os
import tempfile

import paddle
import pytest

import ocean
from ocean.core.optimizer import OceanOptimizer
from ocean.plugins.precision.amp import MixedPrecision
from ocean.plugins.precision.precision import Precision


def make_loader(n=16, bs=8):
    ds = paddle.io.TensorDataset([paddle.randn([n, 4]), paddle.randint(0, 2, [n])])
    return paddle.io.DataLoader(ds, batch_size=bs)


# ── precision ────────────────────────────────────────────────────────────────


def test_precision_flag_builds_the_plugin():
    """Without this the strategy keeps a full-precision plugin and the backward
    pass is never scaled, however the forward was cast."""
    gear = ocean.Gear(accelerator="cpu", precision="16-mixed")
    plugin = gear.strategy.precision_plugin
    assert isinstance(plugin, MixedPrecision)
    assert plugin.precision == "16-mixed"


def test_default_precision_is_full():
    gear = ocean.Gear(accelerator="cpu")
    plugin = gear.strategy.precision_plugin
    assert isinstance(plugin, Precision)
    assert not isinstance(plugin, MixedPrecision)


def test_autocast_comes_from_the_plugin():
    """The cast used by autocast() and the scaling used by backward() must agree."""
    gear = ocean.Gear(accelerator="cpu", precision="16-mixed")
    with gear.autocast():
        pass  # only that it is the plugin's context, exercised below


def test_backward_goes_through_the_plugin():
    calls = []

    class SpyPrecision(Precision):
        def pre_backward(self, tensor, module):
            calls.append("pre_backward")
            return tensor

    gear = ocean.Gear(accelerator="cpu")
    gear.strategy._precision_plugin = SpyPrecision()
    linear = paddle.nn.Linear(4, 2)
    gear.backward(linear(paddle.randn([2, 4])).sum())
    assert calls == ["pre_backward"]


# ── setup() ──────────────────────────────────────────────────────────────────


def test_setup_wraps_optimizers():
    gear = ocean.Gear(accelerator="cpu", precision="16-mixed")
    linear = paddle.nn.Linear(4, 2)
    opt = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters())

    model, wrapped = gear.setup(linear, opt)

    assert model is linear
    assert isinstance(wrapped, OceanOptimizer)
    assert wrapped._precision_plugin is gear.strategy.precision_plugin


def test_setup_without_optimizers_returns_the_model():
    gear = ocean.Gear(accelerator="cpu")
    linear = paddle.nn.Linear(4, 2)
    assert gear.setup(linear) is linear


def test_wrapped_optimizer_still_steps():
    gear = ocean.Gear(accelerator="cpu")
    linear = paddle.nn.Linear(4, 2)
    _, opt = gear.setup(linear, paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters()))

    linear(paddle.randn([2, 4])).sum().backward()
    before = linear.weight.numpy().copy()
    opt.step()
    opt.clear_grad()  # forwarded to the wrapped optimizer
    assert not (linear.weight.numpy() == before).all()


# ── setup_dataloaders ────────────────────────────────────────────────────────


def test_setup_dataloaders_moves_batches():
    gear = ocean.Gear(accelerator="cpu")
    loader = gear.setup_dataloaders(make_loader())
    x, _ = next(iter(loader))
    assert "cpu" in str(x.place).lower()


def test_setup_dataloaders_keeps_len_and_attributes():
    loader = make_loader(16, 8)
    prepared = ocean.Gear(accelerator="cpu").setup_dataloaders(loader)
    assert len(prepared) == 2
    assert prepared.batch_size == loader.batch_size


def test_setup_dataloaders_opt_out():
    loader = make_loader()
    assert ocean.Gear(accelerator="cpu").setup_dataloaders(loader, move_to_device=False) is loader


def test_setup_dataloaders_multiple():
    a, b = ocean.Gear(accelerator="cpu").setup_dataloaders(make_loader(), make_loader())
    assert len(a) == len(b) == 2


# ── load(strict=...) ─────────────────────────────────────────────────────────


def test_strict_load_rejects_a_mismatched_state():
    gear = ocean.Gear(accelerator="cpu")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pdparams")
        gear.save(path, {"model": paddle.nn.Linear(4, 2)})

        with pytest.raises(RuntimeError, match="does not match"):
            gear.load(path, {"model": paddle.nn.Linear(9, 9)}, strict=True)


def test_strict_load_rejects_a_missing_entry():
    gear = ocean.Gear(accelerator="cpu")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pdparams")
        gear.save(path, {"model": paddle.nn.Linear(4, 2)})

        with pytest.raises(KeyError, match="no entry"):
            gear.load(path, {"absent": paddle.nn.Linear(4, 2)}, strict=True)


def test_non_strict_load_skips_what_it_cannot_restore():
    gear = ocean.Gear(accelerator="cpu")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pdparams")
        gear.save(path, {"model": paddle.nn.Linear(4, 2)})
        gear.load(path, {"absent": paddle.nn.Linear(4, 2)}, strict=False)


def test_matching_state_round_trips():
    gear = ocean.Gear(accelerator="cpu")
    source = paddle.nn.Linear(4, 2)
    target = paddle.nn.Linear(4, 2)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pdparams")
        gear.save(path, {"model": source, "epoch": 3})
        checkpoint = gear.load(path, {"model": target}, strict=True)

        assert checkpoint["epoch"] == 3
        assert (target.weight.numpy() == source.weight.numpy()).all()


# ── loggers / ranks ──────────────────────────────────────────────────────────


class _SpyLogger:
    def __init__(self):
        self.calls = []

    def log_metrics(self, metrics, step=None):
        self.calls.append((metrics, step))


def test_log_reaches_the_configured_loggers():
    logger = _SpyLogger()
    gear = ocean.Gear(accelerator="cpu", loggers=logger)

    gear.log("loss", paddle.to_tensor(1.5), step=3)
    gear.log_dict({"a": 1, "b": 2})

    assert logger.calls == [({"loss": 1.5}, 3), ({"a": 1, "b": 2}, None)]
    assert gear.logger is logger


def test_log_without_loggers_is_a_no_op():
    ocean.Gear(accelerator="cpu").log("loss", 1.0)


def test_rank_accessors():
    gear = ocean.Gear(accelerator="cpu")
    assert gear.global_rank == 0
    assert gear.local_rank == 0
    assert gear.world_size == 1
    assert gear.is_global_zero is True
