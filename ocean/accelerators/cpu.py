"""CPU accelerator for PaddlePaddle."""

from typing import Any

import paddle
from ocean.accelerators.accelerator import Accelerator


class CPUAccelerator(Accelerator):
    """CPU accelerator."""

    def setup_device(self, device: Any = None) -> Any:
        return paddle.CPUPlace()

    def setup(self, trainer: Any) -> None:
        paddle.device.set_device("cpu")

    @staticmethod
    def parse_devices(devices: Any) -> int:
        if devices == "auto" or devices is None:
            return 1
        if isinstance(devices, int):
            return max(1, devices)
        return 1

    @staticmethod
    def get_parallel_devices(devices: Any) -> list[Any]:
        return [paddle.CPUPlace()]

    @staticmethod
    def auto_device_count() -> int:
        return 1

    @staticmethod
    def is_available() -> bool:
        return True

    def get_device_stats(self, device: Any) -> dict[str, Any]:
        return {"device": "cpu"}
