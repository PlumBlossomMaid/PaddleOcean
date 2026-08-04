"""Tests that strategy configuration reaches PaddlePaddle.

Covers:
- the ZeRO stage -> sharding level map uses levels Paddle actually accepts
- an invalid sharding level is rejected at construction, not swallowed later
- offload / cpu_offload are passed to group_sharded_parallel
- a failed sharding setup is reported instead of silently training unsharded
- process_group_backend is published where Paddle reads it
- pipeline_parallel_size takes part in the process mesh
"""

import os

import paddle
import pytest

from ocean.plugins.precision.precision import Precision
from ocean.strategies.ddp import DDPStrategy
from ocean.strategies.deepspeed import DeepSpeedStrategy
from ocean.strategies.fsdp import FSDPStrategy
from ocean.strategies.model_parallel import ModelParallelStrategy
from ocean.utils import MisconfigurationException


@pytest.fixture
def sharding_spy(monkeypatch):
    """Capture what the strategies hand to Paddle's sharding entry point."""
    calls = {}

    def spy(model, optimizer, level, **kwargs):
        calls.clear()
        calls.update(level=level, **kwargs)
        return model, optimizer, None

    monkeypatch.setattr(paddle.distributed.sharding, "group_sharded_parallel", spy)
    return calls


def make_fsdp(**kwargs):
    strategy = FSDPStrategy(**kwargs)
    strategy._model = paddle.nn.Linear(2, 2)
    strategy._optimizers = [paddle.optimizer.SGD(learning_rate=0.1, parameters=strategy._model.parameters())]
    strategy._precision_plugin = Precision()
    strategy.setup_optimizers = lambda trainer: None
    return strategy


# ── Sharding levels ──────────────────────────────────────────────────────────


def test_zero_map_uses_levels_paddle_accepts():
    """'os_g2' and 'p_g' are not levels; a call with them raises, and the raise
    was being swallowed, so stages 2 and 3 silently did nothing."""
    assert set(DeepSpeedStrategy.ZERO_MAP.values()) <= set(DeepSpeedStrategy.VALID_LEVELS)
    assert DeepSpeedStrategy.VALID_LEVELS == ("os", "os_g", "p_g_os")


@pytest.mark.parametrize(("stage", "level"), [(1, "os"), (2, "os_g"), (3, "p_g_os")])
def test_zero_stage_maps_to_a_valid_level(stage, level):
    assert DeepSpeedStrategy(zero_stage=stage)._sharding_level == level


@pytest.mark.parametrize("bad", ["p_g", "os_g2", "full"])
def test_invalid_sharding_level_rejected(bad):
    with pytest.raises(MisconfigurationException, match="sharding_level"):
        DeepSpeedStrategy(sharding_level=bad)
    with pytest.raises(MisconfigurationException, match="sharding_level"):
        FSDPStrategy(sharding_level=bad)


def test_invalid_zero_stage_rejected():
    with pytest.raises(MisconfigurationException, match="zero_stage"):
        DeepSpeedStrategy(zero_stage=4)


def test_fsdp_defaults_to_full_sharding():
    assert FSDPStrategy()._sharding_level == "p_g_os"


# ── offload ──────────────────────────────────────────────────────────────────


def test_fsdp_passes_cpu_offload(sharding_spy):
    make_fsdp(cpu_offload=True).setup(None)
    assert sharding_spy == {"level": "p_g_os", "offload": True}


def test_fsdp_offload_defaults_off(sharding_spy):
    make_fsdp().setup(None)
    assert sharding_spy["offload"] is False


def test_deepspeed_passes_offload(sharding_spy):
    strategy = DeepSpeedStrategy(zero_stage=3, offload=True)
    strategy._fleet_initialized = True
    model = paddle.nn.Linear(2, 2)
    strategy._optimizers = [paddle.optimizer.SGD(learning_rate=0.1, parameters=model.parameters())]
    strategy._setup_model(model)
    assert sharding_spy == {"level": "p_g_os", "offload": True}


# ── Failures are reported ────────────────────────────────────────────────────


def test_failed_sharding_is_reported(monkeypatch, capsys):
    """An unsharded run looks exactly like a working one until it runs out of
    memory, so the failure has to be said out loud."""

    def boom(*args, **kwargs):
        raise RuntimeError("no process group")

    monkeypatch.setattr(paddle.distributed.sharding, "group_sharded_parallel", boom)
    make_fsdp().setup(None)

    assert "NOT sharded" in capsys.readouterr().out


def test_failed_deepspeed_sharding_is_reported(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise RuntimeError("no process group")

    monkeypatch.setattr(paddle.distributed.sharding, "group_sharded_parallel", boom)
    strategy = DeepSpeedStrategy()
    strategy._fleet_initialized = True
    model = paddle.nn.Linear(2, 2)
    strategy._optimizers = [paddle.optimizer.SGD(learning_rate=0.1, parameters=model.parameters())]
    assert strategy._setup_model(model) is model
    assert "NOT sharded" in capsys.readouterr().out


# ── process_group_backend ────────────────────────────────────────────────────


def test_process_group_backend_is_published(monkeypatch):
    """Paddle reads the backend from the environment, not from an argument."""
    monkeypatch.delenv("PADDLE_DISTRI_BACKEND", raising=False)
    DDPStrategy(process_group_backend="gloo")._apply_process_group_backend()
    assert os.environ["PADDLE_DISTRI_BACKEND"] == "gloo"


def test_no_backend_leaves_the_environment_alone(monkeypatch):
    monkeypatch.delenv("PADDLE_DISTRI_BACKEND", raising=False)
    DDPStrategy()._apply_process_group_backend()
    assert "PADDLE_DISTRI_BACKEND" not in os.environ


# ── Process mesh ─────────────────────────────────────────────────────────────


def test_pipeline_size_takes_part_in_the_mesh(monkeypatch):
    built = {}

    class FakeMesh:
        def __init__(self, dims):
            built["dims"] = dims

    monkeypatch.setattr(paddle.distributed, "ProcessMesh", FakeMesh)
    monkeypatch.setattr(paddle.distributed, "set_mesh", lambda mesh: None)

    strategy = ModelParallelStrategy(tensor_parallel_size=2, data_parallel_size=2, pipeline_parallel_size=2)
    strategy.setup_optimizers = lambda trainer: None
    strategy._precision_plugin = Precision()
    strategy._model = paddle.nn.Linear(2, 2)
    strategy.setup(None)

    assert built["dims"] == [2, 2, 2]
