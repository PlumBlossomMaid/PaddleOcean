"""Tests for the DDP strategy's collective operations.

These need a process group to run for real, but what the strategy *does with
the results* is checkable here, and that is where the bugs were: a broadcast
that returned each rank its own object, and a mean that was not applied to the
tensor the caller holds.
"""

import paddle
import pytest

from ocean.strategies.ddp import DDPStrategy


@pytest.fixture
def strategy():
    strategy = DDPStrategy()
    strategy._is_initialized = True
    strategy._world_size = 4
    return strategy


# ── broadcast ────────────────────────────────────────────────────────────────


def test_broadcast_of_an_object_returns_the_broadcast_value(strategy, monkeypatch):
    """broadcast_object_list mutates the list and returns None; subscripting
    that return raised, the raise was swallowed, and the caller got its own
    object back."""

    def fake(object_list, src, group=None):
        object_list[0] = "from-rank-0"  # what a real broadcast does

    monkeypatch.setattr(paddle.distributed, "broadcast_object_list", fake)

    assert strategy.broadcast("my-local-value") == "from-rank-0"


def test_broadcast_of_an_object_calls_paddle_once(strategy, monkeypatch):
    calls = []

    def fake(object_list, src, group=None):
        calls.append(src)

    monkeypatch.setattr(paddle.distributed, "broadcast_object_list", fake)
    strategy.broadcast("value", src=2)

    assert calls == [2]


def test_broadcast_of_a_tensor_uses_the_tensor_path(strategy, monkeypatch):
    seen = {}

    def fake(tensor, src, group=None, sync_op=True):
        seen["src"] = src
        paddle.assign(paddle.to_tensor([9.0]), tensor)

    monkeypatch.setattr(paddle.distributed, "broadcast", fake)
    tensor = paddle.to_tensor([1.0])

    assert float(strategy.broadcast(tensor, src=1)[0]) == 9.0
    assert seen["src"] == 1


def test_broadcast_without_a_process_group_is_a_no_op():
    strategy = DDPStrategy()
    strategy._is_initialized = False
    assert strategy.broadcast("value") == "value"


def test_a_failed_broadcast_is_reported(strategy, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise RuntimeError("no group")

    monkeypatch.setattr(paddle.distributed, "broadcast_object_list", boom)
    assert strategy.broadcast("value") == "value"
    assert "rank-local" in capsys.readouterr().out


# ── reduce ───────────────────────────────────────────────────────────────────


def test_mean_reduction_is_applied_in_place(strategy, monkeypatch):
    """all_reduce sums into the caller's tensor; dividing into a *new* one left
    every caller holding the sum."""
    monkeypatch.setattr(paddle.distributed, "all_reduce", lambda tensor, **kwargs: None)
    tensor = paddle.to_tensor([8.0, 4.0])

    returned = strategy.reduce(tensor, "mean")

    assert [float(v) for v in tensor] == [2.0, 1.0]  # world_size 4
    assert [float(v) for v in returned] == [2.0, 1.0]


def test_sum_reduction_leaves_the_value_alone(strategy, monkeypatch):
    monkeypatch.setattr(paddle.distributed, "all_reduce", lambda tensor, **kwargs: None)
    tensor = paddle.to_tensor([8.0])
    strategy.reduce(tensor, "sum")
    assert float(tensor[0]) == 8.0


def test_reduce_without_a_process_group_is_a_no_op():
    strategy = DDPStrategy()
    strategy._is_initialized = False
    tensor = paddle.to_tensor([3.0])
    assert strategy.reduce(tensor, "mean") is tensor
    assert float(tensor[0]) == 3.0


def test_a_failed_reduction_is_reported(strategy, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise RuntimeError("no group")

    monkeypatch.setattr(paddle.distributed, "all_reduce", boom)
    strategy.reduce(paddle.to_tensor([1.0]), "mean")
    assert "rank-local" in capsys.readouterr().out


def test_non_tensor_input_passes_through(strategy):
    assert strategy.reduce("not a tensor", "mean") == "not a tensor"
