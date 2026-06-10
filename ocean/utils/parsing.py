"""Parsing utilities - convert params, flatten dicts, sanitize."""

from typing import Any


def _convert_params(params: Any) -> dict[str, Any]:
    """Convert Namespace or other types to dict."""
    if isinstance(params, dict):
        return params
    if hasattr(params, "__dict__"):
        return vars(params)
    return {"params": params}


def _flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten nested dict with separator."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _sanitize_callable_params(params: dict[str, Any]) -> dict[str, Any]:
    """Replace callable params with their type names."""
    return {k: (str(type(v).__name__) if callable(v) else v) for k, v in params.items()}


def _add_prefix(metrics: dict[str, Any], prefix: str, separator: str) -> dict[str, Any]:
    """Add a prefix to all keys in a dict."""
    if not prefix:
        return metrics
    return {f"{prefix}{separator}{k}": v for k, v in metrics.items()}
