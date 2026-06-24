"""conftest for compat tests — skip on non-CPU devices."""

import paddle


def in_device_blacklist() -> bool:
    """Audio compat tests compare Paddle against librosa (CPU libraries).
    Any non-CPU device (GPU, custom device) may produce different float
    precision, causing spurious failures.  Skip on non-CPU entirely."""
    return "cpu" not in paddle.get_device().lower()


def blacklist_skip_msg() -> str:
    return f"Audio compat tests require CPU (current: {paddle.get_device()})"
