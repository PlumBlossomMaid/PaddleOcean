"""Base precision plugin for paddleOcean."""

from typing import Any

import paddle


class Precision:
    """Base precision plugin - no precision conversion, full float32."""

    precision: str = "32-true"

    def __init__(self, precision: str = "32-true") -> None:
        self.precision = precision

    def convert_module(self, module: paddle.nn.Layer) -> paddle.nn.Layer:
        return module

    def forward_context(self) -> Any:
        import contextlib
        return contextlib.nullcontext()

    def convert_input(self, data: Any) -> Any:
        return data

    def convert_output(self, data: Any) -> Any:
        return data

    def pre_backward(self, tensor: paddle.Tensor, module: Any) -> None: ...
    def backward(self, tensor: paddle.Tensor, model: Any, *args: Any, **kwargs: Any) -> None:
        tensor.backward(*args, **kwargs)
    def post_backward(self, tensor: paddle.Tensor, module: Any) -> paddle.Tensor:
        return tensor.detach()

    def optimizer_step(self, optimizer: paddle.optimizer.Optimizer, **kwargs: Any) -> Any:
        return optimizer.step(**kwargs)

    def unscale_gradients(self, optimizer: paddle.optimizer.Optimizer) -> None: ...

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...
    def teardown(self) -> None: ...
