"""Conditional test skipping utilities.

Analogous to ocean's testing/_runif.py.

Usage::
    @RunIf(min_cuda_version="11.0")
    def test_cuda_feature(): ...

    @skip_on_custom_device(["iluvatar_gpu"])
    def test_float64_audio(): ...
"""

import functools
from typing import Any, Callable, Optional

# Devices known to have limited float64 kernel support.
# These are domestic AI accelerators whose PaddlePaddle plugins may not
# implement all float64 operations (e.g. contiguous, STFT internals).
# We track specific gaps and plan to upstream fixes to PaddlePaddle.
# See: .qwen/iluvatar-unsupported-kernels.md
_UNSUPPORTED_FLOAT64_DEVICES = {
    "iluvatar_gpu",  # 天数智芯 — missing float64 contiguous kernel in complex graphs
}


class RunIf:
    """Decorator to conditionally skip tests.

    Args:
        min_cuda_version: Minimum CUDA version required.
        min_paddle_version: Minimum PaddlePaddle version required.
        skip_if: If True, skip the test.
        reason: Reason for skipping.
    """

    def __init__(
        self,
        min_cuda_version: Optional[str] = None,
        min_paddle_version: Optional[str] = None,
        skip_if: bool = False,
        reason: str = "Skipped by RunIf",
    ) -> None:
        self.min_cuda_version = min_cuda_version
        self.min_paddle_version = min_paddle_version
        self.skip_if = skip_if
        self.reason = reason

    def __call__(self, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if self.skip_if:
                import pytest

                pytest.skip(self.reason)
            if self.min_cuda_version is not None:
                import paddle

                if not paddle.is_compiled_with_cuda():
                    import pytest

                    pytest.skip(f"Requires CUDA >= {self.min_cuda_version}")
            if self.min_paddle_version is not None:
                import paddle
                from packaging.version import Version

                if Version(paddle.__version__) < Version(self.min_paddle_version):
                    import pytest

                    pytest.skip(f"Requires Paddle >= {self.min_paddle_version}")
            return fn(*args, **kwargs)

        return wrapper


def skip_on_custom_device(
    device_types: Optional[list[str]] = None,
    reason: str = "Test incompatible with current custom device",
) -> Callable:
    """Skip a test when running on specific custom (domestic) devices.

    Use this for tests that rely on float64 operations or other features
    not yet implemented in certain PaddlePaddle custom device plugins.

    Args:
        device_types: List of custom device type strings to skip on.
            If ``None`` (default), skip on **any** registered custom device
            (i.e. Iluvatar, Ascend NPU, Cambricon MLU, ...).
        reason: Reason string passed to ``pytest.skip``.
    """
    if device_types is not None:
        _targets = set(device_types)
    else:
        _targets = None

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import paddle

            types = paddle.device.get_all_custom_device_type()
            if not types:
                return fn(*args, **kwargs)

            current = types[0]
            if _targets is None or current in _targets:
                import pytest

                pytest.skip(
                    f"{reason} (device={current})"
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorator
