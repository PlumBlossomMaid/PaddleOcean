"""Checkpoint IO plugins for paddleOcean."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import paddle


class CheckpointIO(ABC):
    """Base class for checkpoint IO."""

    @abstractmethod
    def save_checkpoint(self, checkpoint: dict, path: str, **kwargs: Any) -> None: ...

    @abstractmethod
    def load_checkpoint(self, path: str, **kwargs: Any) -> dict: ...

    @abstractmethod
    def remove_checkpoint(self, path: str) -> None: ...

    def teardown(self) -> None: ...


class PaddleCheckpointIO(CheckpointIO):
    """Checkpoint IO using paddle.save / paddle.load."""

    def save_checkpoint(self, checkpoint: dict, path: str, **kwargs: Any) -> None:
        paddle.save(checkpoint, path)

    def load_checkpoint(self, path: str, **kwargs: Any) -> dict:
        return paddle.load(path)

    def remove_checkpoint(self, path: str) -> None:
        import os
        if os.path.exists(path):
            os.remove(path)
