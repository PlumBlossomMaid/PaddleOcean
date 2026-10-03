"""ROCm accelerator for PaddlePaddle (AMD GPU)."""

from typing import Any

import paddle

from ocean.accelerators.accelerator import Accelerator


class ROCmAccelerator(Accelerator):
    """AMD ROCm GPU accelerator.

    ROCm uses the same CUDAPlace API as NVIDIA (via HIP compatibility layer).
    """

    def setup_device(self, device: Any = None) -> Any:
        """Set the current ROCm device."""
        if isinstance(device, paddle.CUDAPlace):
            idx = device.get_device_id()
        elif isinstance(device, paddle.CPUPlace):
            idx = 0
        else:
            idx = int(device) if device is not None else 0
        paddle.device.set_device(f"gpu:{idx}")
        return paddle.CUDAPlace(idx)

    def setup(self, trainer: Any) -> None:
        if paddle.is_compiled_with_rocm():
            paddle.device.set_device("gpu:0")

    def teardown(self) -> None:
        """Release cached GPU memory on teardown."""
        if paddle.is_compiled_with_rocm():
            paddle.device.cuda.empty_cache()

    @staticmethod
    def parse_devices(devices: Any) -> list[int]:
        """Parse devices into a list of device indices.

        Supports: int (count), str ("0,1,2"), "auto", -1.
        """
        if devices == "auto" or devices is None or devices == -1:
            return list(range(ROCmAccelerator.auto_device_count()))
        if isinstance(devices, int):
            return list(range(devices))
        if isinstance(devices, str):
            parts = devices.split(",")
            return [int(p.strip()) for p in parts if p.strip()]
        return devices if isinstance(devices, list) else [0]

    @staticmethod
    def get_parallel_devices(devices: Any) -> list[Any]:
        devs = ROCmAccelerator.parse_devices(devices)
        return [paddle.CUDAPlace(d) for d in devs]

    @staticmethod
    def auto_device_count() -> int:
        return paddle.device.cuda.device_count() if paddle.is_compiled_with_rocm() else 0

    @staticmethod
    def is_available() -> bool:
        return paddle.is_compiled_with_rocm()

    def get_device_stats(self, device: Any) -> dict[str, Any]:
        if paddle.is_compiled_with_rocm():
            alloc = paddle.device.cuda.memory_allocated()
            reserved = paddle.device.cuda.memory_reserved()
            return {
                "gpu_memory_allocated_mb": alloc / (1024 * 1024),
                "gpu_memory_reserved_mb": reserved / (1024 * 1024),
            }
        return {}
