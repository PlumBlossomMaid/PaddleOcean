"""SingleDeviceStrategy - for single device (CPU or single GPU)."""

from typing import Any, Optional

import paddle

from ocean.strategies.strategy import Strategy
from ocean.accelerators.accelerator import Accelerator
from ocean.plugins.precision import PrecisionPlugin


class SingleDeviceStrategy(Strategy):
    """Strategy for single device execution."""

    def __init__(
        self,
        device: str = "cpu",
        accelerator: Optional[Accelerator] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
    ) -> None:
        super().__init__(accelerator, precision_plugin)
        if device == "cpu" or not paddle.is_compiled_with_cuda():
            self._root_device = paddle.CPUPlace()
        elif device.startswith("gpu"):
            idx = int(device.split(":")[1]) if ":" in device else 0
            self._root_device = paddle.CUDAPlace(idx)
        else:
            self._root_device = paddle.CPUPlace()

    @property
    def root_device(self) -> Any:
        return self._root_device

    @property
    def is_global_zero(self) -> bool:
        return True

    def reduce(self, tensor: Any, reduce_op: str = "mean") -> Any:
        return tensor

    def barrier(self, name: Optional[str] = None) -> None:
        pass

    def broadcast(self, obj: Any, src: int = 0) -> Any:
        return obj
