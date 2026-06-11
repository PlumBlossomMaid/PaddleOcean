"""Half precision plugin - converts model to float16."""

from typing import Any

import paddle

from ocean.plugins.precision.precision import Precision


class HalfPrecision(Precision):
    """Half precision (float16) training."""

    precision: str = "16-true"

    def __init__(self) -> None:
        super().__init__("16-true")

    def convert_module(self, module: paddle.nn.Layer) -> paddle.nn.Layer:
        return module.astype(paddle.float16)

    def forward_context(self) -> Any:
        import contextlib
        return contextlib.nullcontext()
