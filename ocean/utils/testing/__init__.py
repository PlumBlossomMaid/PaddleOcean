"""Conditional test skipping utilities.

Analogous to Lightning's testing/_runif.py.

Usage::
    @RunIf(min_cuda_version="11.0")
    def test_cuda_feature(): ...
"""

import functools
import os
from typing import Any, Callable, Optional, Union


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
