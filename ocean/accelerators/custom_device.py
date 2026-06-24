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

    Supports multi-device training: ``parse_devices(4)`` returns ``[0, 1, 2, 3]``
    and ``get_parallel_devices(4)`` returns ``[CustomPlace(type, 0), ...]``.

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
        """Set the current custom device and return its CustomPlace.

        Both sets the device via ``paddle.device.set_device`` and returns
        the corresponding ``paddle.CustomPlace`` (Lightning-compatible).
        """
        if isinstance(device, paddle.CustomPlace):
            idx = device.get_device_id()
        else:
            idx = int(device) if device is not None else 0
        paddle.device.set_device(f"{self.device_type}:{idx}")
        return paddle.CustomPlace(self.device_type, idx)

    def setup(self, trainer: Any) -> None:
        """Set the default device to the first custom device."""
        paddle.device.set_device(f"{self.device_type}:0")

    @staticmethod
    def parse_devices(devices: Any) -> list[int]:
        """Parse devices into a list of device indices.

        Supports the same formats as ``CUDAAccelerator``:
        - ``"auto"`` / ``-1`` / ``None`` → all available devices
        - ``int`` (e.g. ``4``) → ``[0, 1, 2, 3]``
        - ``str`` (e.g. ``"0,1,2"``) → ``[0, 1, 2]``
        - ``list[int]`` → returned as-is
        """
        if devices == "auto" or devices == -1 or devices is None:
            return list(range(CustomDeviceAccelerator.auto_device_count()))
        if isinstance(devices, int):
            return list(range(devices))
        if isinstance(devices, str):
            parts = devices.split(",")
            return [int(p.strip()) for p in parts if p.strip()]
        return devices

    @staticmethod
    def get_parallel_devices(devices: Any) -> list[Any]:
        """Create a list of ``CustomPlace`` from the parsed device indices."""
        dev_type = CustomDeviceAccelerator._detect_device_type()
        devs = CustomDeviceAccelerator.parse_devices(devices)
        return [paddle.CustomPlace(dev_type, d) for d in devs]

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
