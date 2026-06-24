"""RunIf decorator — Lightning-style conditional test skipping for PaddlePaddle.

Usage::

    from tests.helpers.runif import RunIf

    @RunIf(min_cuda_gpus=2)
    def test_needs_two_gpus():
        ...

    @RunIf(min_cuda_gpus=1, standalone=True)
    def test_needs_one_gpu_and_standalone():
        ...

    @RunIf(skip_windows=True)
    def test_unix_only():
        ...
"""

import os
import sys
from typing import Optional

import paddle
import pytest


def _num_cuda_devices() -> int:
    """Return number of available CUDA devices."""
    try:
        if paddle.is_compiled_with_cuda():
            return paddle.device.cuda.device_count()
    except Exception:
        pass
    return 0


def _is_standalone_test() -> bool:
    """Check if we're in standalone test mode (single-process distributed testing)."""
    return os.environ.get("OCEAN_RUN_STANDALONE_TESTS", "0") == "1"


def _is_on_windows() -> bool:
    return sys.platform == "win32"


class RunIf:
    """Conditional test skip decorator, matching Lightning's ``RunIf``.

    Args:
        min_cuda_gpus: Minimum number of CUDA GPUs required.
        standalone: If True, only runs when ``OCEAN_RUN_STANDALONE_TESTS=1``.
        skip_windows: If True, skips on Windows.
        min_python: Minimum Python version (e.g. ``"3.10"``).
        reason: Custom skip reason (auto-generated if not provided).
    """

    def __init__(
        self,
        min_cuda_gpus: Optional[int] = None,
        standalone: bool = False,
        skip_windows: bool = False,
        min_python: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        self._conditions: list[tuple[bool, str]] = []
        self._reason = reason

        if min_cuda_gpus is not None:
            has_gpus = _num_cuda_devices() >= min_cuda_gpus
            msg = f"requires CUDA GPU count >= {min_cuda_gpus} (found {_num_cuda_devices()})"
            self._conditions.append((not has_gpus, msg))

        if standalone:
            is_standalone = _is_standalone_test()
            self._conditions.append((not is_standalone, "requires OCEAN_RUN_STANDALONE_TESTS=1"))

        if skip_windows:
            self._conditions.append((_is_on_windows(), "not supported on Windows"))

        if min_python is not None:
            actual = f"{sys.version_info.major}.{sys.version_info.minor}"
            too_old = tuple(map(int, min_python.split("."))) > (
                sys.version_info.major,
                sys.version_info.minor,
            )
            self._conditions.append((too_old, f"requires Python >= {min_python} (found {actual})"))

    def __call__(self, obj):
        reasons = [msg for cond, msg in self._conditions if cond]
        if not reasons:
            return obj  # All conditions met — no skip

        skip_reason = self._reason or "; ".join(reasons)
        return pytest.mark.skipif(True, reason=skip_reason)(obj)


# ------------------------------------------------------------------
# pytest_collection_modifyitems hook — filter tests by env variables
#
# Place this in conftest.py to enable env-based filtering:
#
#   OCEAN_RUN_ONLY_CUDA_TESTS=1  → only RunIf(min_cuda_gpus=...) tests
#   OCEAN_RUN_STANDALONE_TESTS=1 → only RunIf(standalone=True) tests
# ------------------------------------------------------------------


def has_runif_marker(item: pytest.Item) -> bool:
    """Check if a test item has any RunIf-generated skipif markers."""
    for marker in item.iter_markers("skipif"):
        if marker.kwargs.get("reason", "").startswith("requires CUDA"):
            return True
        if marker.kwargs.get("reason", "").startswith("requires OCEAN_RUN_STANDALONE"):
            return True
    return False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Filter collected tests based on environment variables.

    Add to your conftest.py::

        from tests.helpers.runif import pytest_collection_modifyitems
    """
    only_cuda = os.environ.get("OCEAN_RUN_ONLY_CUDA_TESTS", "0") == "1"
    only_standalone = os.environ.get("OCEAN_RUN_STANDALONE_TESTS", "0") == "1"

    if not only_cuda and not only_standalone:
        return

    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []

    for item in items:
        if only_cuda and has_runif_marker(item):
            # Only keep tests marked with RunIf(min_cuda_gpus=...)
            selected.append(item)
        elif only_standalone:
            # standalone mode
            standalone = False
            for marker in item.iter_markers("skipif"):
                if "OCEAN_RUN_STANDALONE_TESTS" in marker.kwargs.get("reason", ""):
                    standalone = True
                    break
            if standalone:
                selected.append(item)
            elif only_cuda:
                deselected.append(item)
            else:
                selected.append(item)
        else:
            selected.append(item)

    if deselected:
        items[:] = selected
        config = items[0].session.config if items else pytest.Config
        reporter = config.pluginmanager.get_plugin("terminalreporter")
        if reporter:
            reporter.write_line(
                f"\n[RunIf] Filtered {len(deselected)} tests "
                f"(only_cuda={only_cuda}, only_standalone={only_standalone})\n"
            )
