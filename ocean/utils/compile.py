"""Compile utility - converts dynamic graph to static graph using paddle.jit.to_static.

Equivalent to torch.compile but uses PaddlePaddle's native static graph
compilation which automatically invokes CINN (if available).
"""

from typing import Any, Callable, Optional

import paddle


def to_static(
    model: paddle.nn.Layer,
    input_spec: Optional[list] = None,
    **kwargs: Any,
) -> paddle.nn.Layer:
    """Convert a model from dynamic graph to static graph.

    This triggers PaddlePaddle's CINN compiler for optimization.

    Args:
        model: The dynamic graph model to convert.
        input_spec: Optional input specifications for shape inference.
        **kwargs: Additional arguments to paddle.jit.to_static.

    Returns:
        Static graph version of the model.
    """
    return paddle.jit.to_static(model, input_spec=input_spec, **kwargs)


def is_compiled(model: paddle.nn.Layer) -> bool:
    """Check if a model is compiled to static graph.

    Args:
        model: The model to check.

    Returns:
        True if the model is a static graph model.
    """
    return hasattr(model, "forward") and hasattr(model, "program")


def compile(
    model: paddle.nn.Layer,
    full_graph: bool = True,
    **kwargs: Any,
) -> paddle.nn.Layer:
    """Compile a model using PaddlePaddle's static graph (CINN).

    Analogous to torch.compile().

    Args:
        model: The model to compile.
        full_graph: If True, compile the entire graph.
        **kwargs: Additional arguments to paddle.jit.to_static.

    Returns:
        Compiled static graph model.
    """
    return to_static(model, **kwargs)
