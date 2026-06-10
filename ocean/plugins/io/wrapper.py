"""Checkpoint IO wrapper for combining multiple IO strategies."""

from typing import Any, Optional

from ocean.plugins.io import CheckpointIO, PaddleCheckpointIO


class WrapperCheckpointIO(CheckpointIO):
    """Wrapper that combines multiple CheckpointIO strategies.

    Args:
        base_io: Primary IO strategy (default: PaddleCheckpointIO).
    """

    def __init__(self, base_io: Optional[CheckpointIO] = None) -> None:
        self._base_io = base_io or PaddleCheckpointIO()

    def save_checkpoint(self, checkpoint: dict, path: str, **kwargs: Any) -> None:
        self._base_io.save_checkpoint(checkpoint, path, **kwargs)

    def load_checkpoint(self, path: str, **kwargs: Any) -> dict:
        return self._base_io.load_checkpoint(path, **kwargs)

    def remove_checkpoint(self, path: str) -> None:
        self._base_io.remove_checkpoint(path)
