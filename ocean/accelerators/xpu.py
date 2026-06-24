"""XPU (Baidu Kunlun) accelerator for PaddlePaddle.

XPU is hardcoded in the PaddlePaddle main framework — it is NOT a
CustomDevice plugin.  Detection uses ``paddle.is_compiled_with_xpu()``
and devices are placed via ``paddle.XPUPlace()``.

Supported hardware:
    - Kunlunxin P800 (HBM2e, 32 GB)
    - Older Kunlun R200/R300 series

Reference:
    https://www.paddlepaddle.org.cn/documentation/docs/zh/hardware_support/xpu/xpu-p800_paddle_tutorial_cn.html
"""

from typing import Any

import paddle

from ocean.accelerators.accelerator import Accelerator


class XPUAccelerator(Accelerator):
    """Baidu Kunlunxin XPU accelerator.

    XPU differs from other domestic AI chips in that it is compiled
    directly into the PaddlePaddle core, not loaded as a CustomDevice
    plugin.  Therefore ``paddle.device.get_all_custom_device_type()``
    does **not** report ``"xpu"``.

    Supports multi-device training: when multiple XPU cards are available,
    ``parse_devices(4)`` returns ``[0, 1, 2, 3]`` and
    ``get_parallel_devices(4)`` returns ``[XPUPlace(0), XPUPlace(1), ...]``.

    Usage::

        accelerator = XPUAccelerator()
        model, optimizers = accelerator.setup(trainer)
    """

    def setup_device(self, device: Any = None) -> Any:
        """Set the current XPU device and return its XPUPlace.

        Both sets the device via ``paddle.device.set_device`` and returns
        the corresponding ``paddle.XPUPlace`` (Lightning-compatible).
        """
        if isinstance(device, paddle.XPUPlace):
            idx = device.get_device_id()
        else:
            idx = int(device) if device is not None else 0
        paddle.device.set_device(f"xpu:{idx}")
        return paddle.XPUPlace(idx)

    def setup(self, trainer: Any) -> None:
        if paddle.is_compiled_with_xpu():
            paddle.device.set_device("xpu:0")

    def teardown(self) -> None:
        pass

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
            return list(range(XPUAccelerator.auto_device_count()))
        if isinstance(devices, int):
            return list(range(devices))
        if isinstance(devices, str):
            parts = devices.split(",")
            return [int(p.strip()) for p in parts if p.strip()]
        return devices

    @staticmethod
    def get_parallel_devices(devices: Any) -> list[Any]:
        """Create a list of ``XPUPlace`` from the parsed device indices."""
        devs = XPUAccelerator.parse_devices(devices)
        return [paddle.XPUPlace(d) for d in devs]

    @staticmethod
    def auto_device_count() -> int:
        if paddle.is_compiled_with_xpu():
            try:
                return paddle.device.device_count("xpu")
            except Exception:
                return 1
        return 0

    @staticmethod
    def is_available() -> bool:
        return paddle.is_compiled_with_xpu()
