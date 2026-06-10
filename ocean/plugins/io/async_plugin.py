"""Async checkpoint IO using thread pool."""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from ocean.plugins.io import PaddleCheckpointIO


class AsyncCheckpointIO(PaddleCheckpointIO):
    """Checkpoint IO with async saving.

    Uses a thread pool to save checkpoints in background.

    Args:
        max_workers: Max threads for async saving.
    """

    def __init__(self, max_workers: int = 2) -> None:
        super().__init__()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()

    def save_checkpoint(self, checkpoint: dict, path: str, **kwargs: Any) -> None:
        self._executor.submit(super().save_checkpoint, checkpoint, path, **kwargs)

    def teardown(self) -> None:
        self._executor.shutdown(wait=True)
