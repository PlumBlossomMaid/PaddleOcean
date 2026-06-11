"""IPU (Graphcore) accelerator for PaddlePaddle."""

from typing import Any

import paddle

from ocean.accelerators.accelerator import Accelerator


class IPUAccelerator(Accelerator):
    """Graphcore IPU accelerator."""

    def setup_device(self, device: Any = None) -> Any:
        return paddle.IPUPlace()

    @staticmethod
    def parse_devices(devices: Any) -> list[int]:
        return [0]

    @staticmethod
    def get_parallel_devices(devices: Any) -> list[Any]:
        return [paddle.IPUPlace()]

    @staticmethod
    def auto_device_count() -> int:
        return 1 if paddle.is_compiled_with_ipu() else 0

    @staticmethod
    def is_available() -> bool:
        return paddle.is_compiled_with_ipu()
