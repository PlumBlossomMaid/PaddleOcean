"""Memory utility functions."""

from typing import Any, Optional

import paddle


def is_cuda_memory_available() -> bool:
    """Check if CUDA is available for GPU memory operations."""
    return paddle.is_compiled_with_cuda()


def get_gpu_memory() -> dict[str, float]:
    """Get GPU memory usage in MB."""
    if not paddle.is_compiled_with_cuda():
        return {}
    return {
        "allocated_mb": paddle.device.cuda.memory_allocated() / (1024 * 1024),
        "reserved_mb": paddle.device.cuda.memory_reserved() / (1024 * 1024),
    }
