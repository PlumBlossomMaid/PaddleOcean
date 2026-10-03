"""Wrapper for combining multiple IO strategies."""

from typing import Any, Optional


class WrapperCheckpointIO:
    """Wrapper that combines multiple CheckpointIO strategies.

    Args:
        base_io: Primary IO strategy (default: PaddleCheckpointIO).
    """

    def __init__(self, base_io: Optional[Any] = None) -> None:
        if base_io is None:
            from ocean.plugins.io import PaddleCheckpointIO

            base_io = PaddleCheckpointIO()
        self._base_io = base_io

    def save_checkpoint(self, checkpoint: dict, path: str, **kwargs: Any) -> None:
        self._base_io.save_checkpoint(checkpoint, path, **kwargs)

    def load_checkpoint(self, path: str, **kwargs: Any) -> dict:
        return self._base_io.load_checkpoint(path, **kwargs)

    def remove_checkpoint(self, path: str) -> None:
        self._base_io.remove_checkpoint(path)

    def teardown(self) -> None:
        if hasattr(self._base_io, "teardown"):
            self._base_io.teardown()
