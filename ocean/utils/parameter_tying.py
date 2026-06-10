"""Parameter tying detection utilities."""

from typing import Any, Optional

import paddle


def find_tying_parameters(model: paddle.nn.Layer) -> list[tuple[str, str]]:
    """Find all tied (shared) parameters in a model.

    Tied parameters are those that share the same tensor memory.

    Args:
        model: The model to check.

    Returns:
        List of (name1, name2) tuples for tied parameter pairs.
    """
    param_to_names: dict[int, list[str]] = {}
    for name, param in model.named_parameters():
        addr = id(param)
        if addr not in param_to_names:
            param_to_names[addr] = []
        param_to_names[addr].append(name)

    return [(names[0], names[1]) for names in param_to_names.values() if len(names) > 1]


def assert_no_tying_parameters(model: paddle.nn.Layer) -> None:
    """Assert that the model has no tied parameters.

    Raises:
        ValueError: If tied parameters are found.
    """
    tied = find_tying_parameters(model)
    if tied:
        msg = "\n".join(f"{a} <-> {b}" for a, b in tied)
        raise ValueError(f"Found tied parameters:\n{msg}")
