"""Rank zero utilities — Lightning-style functions for distributed training.

Usage::

    @rank_zero_only
    def log_metrics(self, ...): ...   # only runs on rank 0

    rank_zero_info("hello")            # prints only on rank 0
    rank_zero_warn("careful")          # warns only on rank 0

The strategy sets ``rank_zero_only.rank`` during ``setup_environment()``
so that the decorator works correctly in DDP mode.
"""

import os
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class _RankZeroOnly:
    """Callable decorator that skips execution on non-zero ranks.

    The strategy sets ``rank_zero_only.rank`` (mutable attribute) during
    ``setup_environment()`` so the check reflects the real distributed rank.
    Falls back to ``LOCAL_RANK`` env var before initialization.
    """

    rank: int = 0

    def __call__(self, fn: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if self.rank > 0:
                return None
            return fn(*args, **kwargs)

        return wrapper  # type: ignore


rank_zero_only = _RankZeroOnly()


def _refresh_rank() -> None:
    """Update ``rank_zero_only.rank`` from environment (used by strategy)."""
    rank_zero_only.rank = int(os.environ.get("LOCAL_RANK", 0))


# ------------------------------------------------------------------
# Convenience logging functions (Lightning-style)
# ------------------------------------------------------------------


@rank_zero_only
def rank_zero_debug(msg: str) -> None:
    print(f"[DEBUG] {msg}")


@rank_zero_only
def rank_zero_info(msg: str) -> None:
    print(f"[INFO] {msg}")


@rank_zero_only
def rank_zero_warn(msg: str) -> None:
    print(f"[WARN] {msg}")


# ------------------------------------------------------------------
# _DummyExperiment — no-op for non-rank-0 processes (Lightning compat)
# ------------------------------------------------------------------


class _DummyExperiment:
    """Stand-in for the real experiment on non-zero ranks (all methods are no-ops)."""

    def __getattr__(self, name: str) -> Any:
        return _no_op

    def __enter__(self):
        return self

    def __exit__(self, *args: Any) -> None:
        pass


def _no_op(*args: Any, **kwargs: Any) -> "_DummyExperiment":
    return _DummyExperiment()


def rank_zero_experiment(fn: Callable) -> Callable:
    """Decorator for the ``experiment`` property — returns ``_DummyExperiment`` on ranks > 0.

    Usage::

        @property
        @rank_zero_experiment
        def experiment(self): ...
    """

    def wrapper(self: Any) -> Any:
        if rank_zero_only.rank > 0:
            return _DummyExperiment()
        return fn(self)

    return wrapper


# ------------------------------------------------------------------
# WarningCache (Lightning-style)
# ------------------------------------------------------------------


class WarningCache:
    """Cache warnings to avoid repeating the same warning."""

    def __init__(self) -> None:
        self._warnings: set[str] = set()

    def warn(self, msg: str) -> None:
        if msg not in self._warnings:
            self._warnings.add(msg)
            rank_zero_warn(msg)
