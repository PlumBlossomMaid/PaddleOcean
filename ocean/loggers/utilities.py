"""Logger utilities - helper functions for loggers."""

from typing import Any


def _convert_json_serializable(value: Any) -> Any:
    """Convert a value to JSON-serializable type."""
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_convert_json_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _convert_json_serializable(v) for k, v in value.items()}
    return str(value)
