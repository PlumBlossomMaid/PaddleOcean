"""CustomDevice accelerator for PaddlePaddle (e.g., Ascend NPU)."""

from typing import Any

import paddle
from ocean.accelerators.accelerator import Accelerator


class CustomDeviceAccelerator(Accelerator):
    """Custom device accelerator (e.g., Ascend NPU, Cambricon MLU).

    Args:
        device_type: Custom device type string (e.g., 'npu', 'mlu').
    """

    def __init__(self, device_type: str = "custom") -> None:
        self.device_type = device_type

    def setup_device(self, device: Any = None) -> Any:
        return paddle.CustomPlace(self.device_type, 0)

    @staticmethod
    def parse_devices(devices: Any) -> list[int]:
        return [0]

    @staticmethod
    def get_parallel_devices(devices: Any) -> list[Any]:
        return [paddle.CustomPlace("custom", 0)]

    @staticmethod
    def auto_device_count() -> int:
        return 1 if paddle.is_compiled_with_custom_device("custom") else 0

    @staticmethod
    def is_available() -> bool:
        return paddle.is_compiled_with_custom_device("custom")
