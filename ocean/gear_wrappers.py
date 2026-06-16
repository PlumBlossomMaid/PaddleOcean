"""Wrappers for Gear - _FabricModule and _FabricOptimizer.

Analogous to paddleOcean Fabric's wrappers.py.
Wraps model forward/backward and optimizer step with precision/ddp handling.
"""

from typing import Any

import paddle


class _FabricModule(paddle.nn.Layer):
    """Wrapper around a model that handles precision conversion, device placement,
    and forward context automatically.

    When forward() is called, it:
    1. Converts inputs to the correct precision
    2. Runs the forward pass in the precision context
    3. Converts outputs back
    """

    def __init__(self, module: paddle.nn.Layer, precision_plugin: Any = None) -> None:
        super().__init__()
        self._module = module
        self._precision_plugin = precision_plugin
        self._forward_methods: set[str] = set()

    @property
    def module(self) -> paddle.nn.Layer:
        return self._module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self._precision_plugin is not None:
            args = self._precision_plugin.convert_input(args)
            kwargs = {k: self._precision_plugin.convert_input(v) for k, v in kwargs.items()}
            with self._precision_plugin.forward_context():
                output = self._module(*args, **kwargs)
            return self._precision_plugin.convert_output(output)
        return self._module(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._module, name)

    def state_dict(self, *args: Any, **kwargs: Any) -> dict:
        return self._module.state_dict(*args, **kwargs)

    def set_state_dict(self, *args: Any, **kwargs: Any) -> None:
        self._module.set_state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        if strict:
            self._module.set_state_dict(state_dict)
        else:
            self._module.set_dict(state_dict)


class _FabricOptimizer:
    """Wrapper around an optimizer that delegates step/state_dict to the strategy."""

    def __init__(self, optimizer: paddle.optimizer.Optimizer, strategy: Any = None) -> None:
        self._optimizer = optimizer
        self._strategy = strategy

    @property
    def optimizer(self) -> paddle.optimizer.Optimizer:
        return self._optimizer

    def step(self, *args: Any, **kwargs: Any) -> Any:
        if self._strategy is not None:
            return self._strategy.optimizer_step(self._optimizer, *args, **kwargs)
        return self._optimizer.step(*args, **kwargs)

    def state_dict(self) -> dict:
        return self._optimizer.state_dict()

    def set_state_dict(self, state_dict: dict) -> None:
        self._optimizer.set_state_dict(state_dict)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._optimizer, name)
