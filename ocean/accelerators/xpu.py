"""XPU (Baidu Kunlun) accelerator for PaddlePaddle."""

from typing import Any

import paddle

from ocean.accelerators.accelerator import Accelerator


class XPUAccelerator(Accelerator):
    """Baidu Kunlun XPU accelerator."""

    def setup_device(self, device: Any = None) -> Any:
        return paddle.XPUPlace(0)

    @staticmethod
    def parse_devices(devices: Any) -> list[int]:
        if devices == "auto" or devices is None:
            return [0]
        return [0]

    @staticmethod
    def get_parallel_devices(devices: Any) -> list[Any]:
        return [paddle.XPUPlace(0)]

    @staticmethod
    def auto_device_count() -> int:
        return 1 if paddle.is_compiled_with_xpu() else 0

    @staticmethod
    def is_available() -> bool:
        return paddle.is_compiled_with_xpu()
