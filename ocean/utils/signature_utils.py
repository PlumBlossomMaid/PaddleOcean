"""Signature utilities - inspect function signatures for hooks."""

import inspect
from typing import Any, Callable, Optional


def is_param_in_hook_signature(
    hook_fn: Callable,
    param_name: str,
    explicit: bool = False,
) -> bool:
    """Check if a parameter name exists in a hook's signature.

    Args:
        hook_fn: The hook function to inspect.
        param_name: The parameter name to check for.
        explicit: If True, only return True if the parameter is explicitly named.

    Returns:
        True if the parameter is in the signature.
    """
    try:
        sig = inspect.signature(hook_fn)
        return param_name in sig.parameters
    except (ValueError, TypeError):
        return False
