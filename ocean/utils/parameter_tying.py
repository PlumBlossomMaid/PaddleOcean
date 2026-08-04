"""Parameter tying detection utilities."""

import paddle


def find_tying_parameters(model: paddle.nn.Layer) -> list[tuple[str, str]]:
    """Find all tied (shared) parameters in a model.

    Tied parameters are those that share the same tensor memory.

    Args:
        model: The model to check.

    Returns:
        List of ``(name, other_name)`` tuples, one per extra name a shared
        parameter is reachable under.
    """
    # ``remove_duplicate=False`` is the whole point: the default de-duplicates
    # by tensor, so a shared parameter is listed once under a single name and
    # the very thing being looked for can never appear.
    param_to_names: dict[int, list[str]] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        param_to_names.setdefault(id(param), []).append(name)

    tied: list[tuple[str, str]] = []
    for names in param_to_names.values():
        first = names[0]
        tied.extend((first, other) for other in names[1:])
    return tied


def assert_no_tying_parameters(model: paddle.nn.Layer) -> None:
    """Assert that the model has no tied parameters.

    Raises:
        ValueError: If tied parameters are found.
    """
    tied = find_tying_parameters(model)
    if tied:
        msg = "\n".join(f"{a} <-> {b}" for a, b in tied)
        raise ValueError(f"Found tied parameters:\n{msg}")
