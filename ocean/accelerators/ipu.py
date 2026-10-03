"""IPU (Graphcore) accelerator for PaddlePaddle."""

from typing import Any

import paddle

from ocean.accelerators.accelerator import Accelerator


class IPUAccelerator(Accelerator):
    """Graphcore IPU accelerator.

    PaddlePaddle supports IPU via ``paddle.device.set_device('ipu')``
    and ``paddle.IPUPlace()``.
    """

    def setup_device(self, device: Any = None) -> Any:
        """Set the current IPU device and return IPUPlace."""
        paddle.device.set_device("ipu")
        return paddle.IPUPlace()

    def setup(self, trainer: Any) -> None:
        if paddle.is_compiled_with_ipu():
            paddle.device.set_device("ipu")

    def teardown(self) -> None:
        """IPU teardown — no cache to clear, but reset device if needed."""
        pass

    @staticmethod
    def parse_devices(devices: Any) -> list[int]:
        """Parse devices into a list of device indices.

        Supports: int (count), str ("0,1,2"), "auto".
        """
        if devices == "auto" or devices is None or devices == -1:
            return list(range(IPUAccelerator.auto_device_count()))
        if isinstance(devices, int):
            return list(range(devices))
        if isinstance(devices, str):
            parts = devices.split(",")
            return [int(p.strip()) for p in parts if p.strip()]
        return devices if isinstance(devices, list) else [0]

    @staticmethod
    def get_parallel_devices(devices: Any) -> list[Any]:
        devs = IPUAccelerator.parse_devices(devices)
        return [paddle.IPUPlace() for _ in devs]

    @staticmethod
    def auto_device_count() -> int:
        return 1 if paddle.is_compiled_with_ipu() else 0

    @staticmethod
    def is_available() -> bool:
        return paddle.is_compiled_with_ipu()

    def get_device_stats(self, device: Any) -> dict[str, Any]:
        # IPU does not expose memory stats via PaddlePaddle API
        return {}
