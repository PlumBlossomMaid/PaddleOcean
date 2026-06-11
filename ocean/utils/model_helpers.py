"""Model helpers for paddleOcean."""

from typing import Any


def _restricted_classmethod(func: Any) -> Any:
    """Decorator for classmethods that should raise an informative error."""
    import functools
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)
    return classmethod(wrapper)
