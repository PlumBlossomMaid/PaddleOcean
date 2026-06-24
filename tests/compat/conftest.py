"""conftest for compat tests — skip on non-CPU devices.

Audio compat tests compare Paddle against librosa (CPU library).
Float precision differences and missing kernels (e.g. complex128
on XPU, contiguous on custom devices) cause spurious failures on
any non-CPU device.  These tests are copied from PaddlePaddle
upstream — even upstream does not guarantee they all pass on
non-CPU devices.
"""

import paddle
import pytest

if "cpu" not in paddle.get_device().lower():
    pytest.skip(
        f"Compat tests require CPU (current: {paddle.get_device()})",
        allow_module_level=True,
    )


def in_device_blacklist() -> bool:
    """Return True if the current device is not CPU.

    Audio compat tests compare Paddle against librosa (CPU libraries).
    Any non-CPU device (GPU, custom device) may produce different float
    precision, causing spurious failures.
    """
    return "cpu" not in paddle.get_device().lower()


def blacklist_skip_msg() -> str:
    return f"Audio compat tests require CPU (current: {paddle.get_device()})"
