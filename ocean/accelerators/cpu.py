"""CPU accelerator for PaddlePaddle."""

from typing import Any

import paddle

from ocean.accelerators.accelerator import Accelerator


class CPUAccelerator(Accelerator):
    """CPU accelerator.

    Matches Lightning's pattern: ``get_parallel_devices(N)`` returns
    ``[CPUPlace(), CPUPlace(), ...]`` with ``N`` entries so the
    connector can create ``DDPStrategy`` for multi-CPU distributed runs.
    """

    def setup_device(self, device: Any = None) -> Any:
        return paddle.CPUPlace()

    def setup(self, trainer: Any) -> None:
        paddle.device.set_device("cpu")

    @staticmethod
    def parse_devices(devices: Any) -> list[int]:
        """Parse devices into a list of indices (always ``[0]*N`` for CPU).

        Returns a list so the return type is consistent with other accelerators.
        """
        if devices == "auto" or devices is None:
            return [0]
        if isinstance(devices, int):
            return [0] * max(1, devices)
        if isinstance(devices, str):
            parts = devices.split(",")
            return [0] * max(1, len([p for p in parts if p.strip()]))
        return devices

    @staticmethod
    def get_parallel_devices(devices: Any) -> list[Any]:
        """Return a list of CPU devices. Length is determined by ``parse_devices``."""
        devs = CPUAccelerator.parse_devices(devices)
        return [paddle.CPUPlace() for _ in devs]

    @staticmethod
    def auto_device_count() -> int:
        return 1

    @staticmethod
    def is_available() -> bool:
        return True

    def get_device_stats(self, device: Any) -> dict[str, Any]:
        return {"device": "cpu"}
