"""Rank zero utilities - decorators and functions for distributed training."""

import os
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def rank_zero_only(fn: F) -> F:
    """Decorator to run a function only on rank 0.

    In non-distributed mode (world_size=1), the function always runs.
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            return fn(*args, **kwargs)
        return None
    return wrapper  # type: ignore


def rank_zero_debug(msg: str) -> None:
    """Print debug message only on rank 0."""
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"[DEBUG] {msg}")


def rank_zero_info(msg: str) -> None:
    """Print info message only on rank 0."""
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"[INFO] {msg}")


def rank_zero_warn(msg: str) -> None:
    """Print warning message only on rank 0."""
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"[WARN] {msg}")


class WarningCache:
    """Cache warnings to avoid repeating them."""

    def __init__(self) -> None:
        self._warnings: set[str] = set()

    def warn(self, msg: str) -> None:
        if msg not in self._warnings:
            self._warnings.add(msg)
            rank_zero_warn(msg)
