"""Import utilities for optional dependencies."""



def _module_available(module_path: str) -> bool:
    """Check if a module is available without importing it."""
    try:
        import importlib
        return importlib.util.find_spec(module_path) is not None
    except Exception:
        return False


def _compare_version(version: str, target: str, op: str = ">=") -> bool:
    """Compare two version strings."""
    try:
        from packaging.version import Version
        return op_mapping[op](Version(version), Version(target))
    except Exception:
        return False


op_mapping = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
}
