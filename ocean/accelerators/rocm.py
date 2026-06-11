"""ROCm accelerator for PaddlePaddle (AMD GPU)."""

from typing import Any

import paddle

from ocean.accelerators.accelerator import Accelerator


class ROCmAccelerator(Accelerator):
    """AMD ROCm GPU accelerator."""

    def setup_device(self, device: Any = None) -> Any:
        return paddle.CUDAPlace(0)  # ROCm uses CUDAPlace

    @staticmethod
    def parse_devices(devices: Any) -> list[int]:
        if devices == "auto" or devices is None:
            return [0]
        if isinstance(devices, int):
            return list(range(devices))
        return [0]

    @staticmethod
    def get_parallel_devices(devices: Any) -> list[Any]:
        return [paddle.CUDAPlace(0)]

    @staticmethod
    def auto_device_count() -> int:
        return 1 if paddle.is_compiled_with_rocm() else 0

    @staticmethod
    def is_available() -> bool:
        return paddle.is_compiled_with_rocm()
