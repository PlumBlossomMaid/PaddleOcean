"""CUDA (GPU) accelerator for PaddlePaddle."""

from typing import Any

import paddle

from ocean.accelerators.accelerator import Accelerator


class CUDAAccelerator(Accelerator):
    """NVIDIA GPU (CUDA) accelerator."""

    def setup_device(self, device: Any = None) -> Any:
        """Set the current CUDA device (Lightning-compatible).

        Both sets the device via ``paddle.device.set_device`` and returns
        the corresponding ``CUDAPlace``.
        """
        if isinstance(device, paddle.CUDAPlace):
            idx = device.get_device_id()
        else:
            idx = int(device) if device is not None else 0
        paddle.device.set_device(f"gpu:{idx}")
        return paddle.CUDAPlace(idx)

    def setup(self, trainer: Any) -> None:
        if paddle.is_compiled_with_cuda():
            paddle.device.set_device("gpu:0")

    def teardown(self) -> None:
        pass

    @staticmethod
    def parse_devices(devices: Any) -> list[int]:
        if devices == "auto" or devices == -1 or devices is None:
            return list(range(CUDAAccelerator.auto_device_count()))
        if isinstance(devices, int):
            return list(range(devices))
        if isinstance(devices, str):
            parts = devices.split(",")
            return [int(p.strip()) for p in parts if p.strip()]
        return devices

    @staticmethod
    def get_parallel_devices(devices: Any) -> list[Any]:
        devs = CUDAAccelerator.parse_devices(devices)
        return [paddle.CUDAPlace(d) for d in devs]

    @staticmethod
    def auto_device_count() -> int:
        if paddle.is_compiled_with_cuda():
            return paddle.device.cuda.device_count()
        return 0

    @staticmethod
    def is_available() -> bool:
        return paddle.is_compiled_with_cuda()

    def get_device_stats(self, device: Any) -> dict[str, Any]:
        if paddle.is_compiled_with_cuda():
            alloc = paddle.device.cuda.memory_allocated()
            reserved = paddle.device.cuda.memory_reserved()
            return {
                "gpu_memory_allocated_mb": alloc / (1024 * 1024),
                "gpu_memory_reserved_mb": reserved / (1024 * 1024),
            }
        return {}


GPUAccelerator = CUDAAccelerator
