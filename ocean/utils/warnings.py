"""Warning utilities for paddleOcean.

``rank_zero_warn`` and ``WarningCache`` are re-exported from
``ocean.utils.rank_zero`` — this module exists for backward compat.
"""

from ocean.utils.rank_zero import rank_zero_warn  # noqa: F401


class WarningCache:
    """Cache warnings to avoid repeating the same warning."""

    def __init__(self) -> None:
        self._warnings: set[str] = set()

    def warn(self, msg: str) -> None:
        if msg not in self._warnings:
            self._warnings.add(msg)
            rank_zero_warn(msg)
