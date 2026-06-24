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

    Usage::

        accelerator = XPUAccelerator()
        model, optimizers = accelerator.setup(trainer)
    """

    def setup_device(self, device: Any = None) -> Any:
        return paddle.XPUPlace(0)

    def setup(self, trainer: Any) -> None:
        if paddle.is_compiled_with_xpu():
            paddle.device.set_device("xpu:0")

    def teardown(self) -> None:
        pass

    @staticmethod
    def parse_devices(devices: Any) -> list[int]:
        if devices == "auto" or devices is None:
            return [0]
        return [0]

    @staticmethod
    def get_parallel_devices(devices: Any) -> list[Any]:
        return [paddle.XPUPlace(0)]

    @staticmethod
    def auto_device_count() -> int:
        return 1 if paddle.is_compiled_with_xpu() else 0

    @staticmethod
    def is_available() -> bool:
        return paddle.is_compiled_with_xpu()
