"""Async checkpoint IO using thread pool."""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class AsyncCheckpointIO:
    """Checkpoint IO with async saving.

    Uses a thread pool to save checkpoints in background.

    Args:
        max_workers: Max threads for async saving.
    """

    def __init__(self, max_workers: int = 2) -> None:
        from ocean.plugins.io import PaddleCheckpointIO

        self._base_io = PaddleCheckpointIO()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()

    def save_checkpoint(self, checkpoint: dict, path: str, **kwargs: Any) -> None:
        self._executor.submit(self._base_io.save_checkpoint, checkpoint, path, **kwargs)

    def load_checkpoint(self, path: str, **kwargs: Any) -> dict:
        return self._base_io.load_checkpoint(path, **kwargs)

    def remove_checkpoint(self, path: str) -> None:
        self._base_io.remove_checkpoint(path)

    def teardown(self) -> None:
        self._executor.shutdown(wait=True)
