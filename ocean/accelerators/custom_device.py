"""CustomDevice accelerator for PaddlePaddle (Iluvatar, Ascend NPU, Cambricon MLU, ...).

Auto-detects the available custom device type via PaddlePaddle's native
``get_all_custom_device_type()`` API — no hardcoded vendor names needed.
"""

from typing import Any, Optional

import paddle

from ocean.accelerators.accelerator import Accelerator


class CustomDeviceAccelerator(Accelerator):
    """Custom device accelerator for domestic AI chips.

    Automatically detects the available device type from the runtime via
    ``paddle.device.get_all_custom_device_type()``, so it works seamlessly
    with Iluvatar, Ascend NPU, Cambricon MLU, and any PaddlePaddle-compatible
    custom device — no manual configuration needed.

    Args:
        device_type: Optional explicit device type string (e.g. ``'npu'``).
            If ``None`` (default), auto-detect from the runtime.
    """

    def __init__(self, device_type: Optional[str] = None) -> None:
        self.device_type = device_type or self._detect_device_type()

    # ------------------------------------------------------------------
    # Auto-detection — single source of truth: Paddle's native API
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device_type() -> str:
        """Return the first registered custom device type from Paddle runtime."""
        types = paddle.device.get_all_custom_device_type()
        if types:
            return types[0]
        raise RuntimeError(
            "No PaddlePaddle custom device plugin detected. "
            "Ensure the appropriate package is installed "
            "(e.g. paddle-iluvatar-gpu, paddle-npu)."
        )

    # ------------------------------------------------------------------
    # Accelerator interface
    # ------------------------------------------------------------------

    def setup_device(self, device: Any = None) -> Any:
        return paddle.CustomPlace(self.device_type, 0)

    @staticmethod
    def parse_devices(devices: Any) -> list[int]:
        return [0]

    @staticmethod
    def get_parallel_devices(devices: Any) -> list[Any]:
        dev_type = CustomDeviceAccelerator._detect_device_type()
        return [paddle.CustomPlace(dev_type, 0)]

    @staticmethod
    def auto_device_count() -> int:
        try:
            dev_type = CustomDeviceAccelerator._detect_device_type()
            return paddle.device.device_count(dev_type)
        except RuntimeError:
            return 0

    @staticmethod
    def is_available() -> bool:
        return bool(paddle.device.get_all_custom_device_type())
