"""Warning utilities for paddleOcean."""

import warnings


def rank_zero_warn(msg: str) -> None:
    """Emit a warning only on rank 0."""
    import os
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        warnings.warn(msg, stacklevel=2)


class WarningCache:
    """Cache warnings to avoid repeating the same warning."""

    def __init__(self) -> None:
        self._warnings: set[str] = set()

    def warn(self, msg: str) -> None:
        if msg not in self._warnings:
            self._warnings.add(msg)
            rank_zero_warn(msg)
