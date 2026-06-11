"""Double precision plugin - converts model to float64."""


import paddle

from ocean.plugins.precision.precision import Precision


class DoublePrecision(Precision):
    """Double precision (float64) training."""

    precision: str = "64-true"

    def __init__(self) -> None:
        super().__init__("64-true")

    def convert_module(self, module: paddle.nn.Layer) -> paddle.nn.Layer:
        return module.astype(paddle.float64)
