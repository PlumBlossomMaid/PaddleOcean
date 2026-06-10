"""Gradient utilities for paddleOcean."""

from typing import Any, Optional

import paddle


def clip_gradients(
    model: paddle.nn.Layer,
    clip_val: Optional[float] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
) -> Optional[paddle.Tensor]:
    """Clip gradients of a model.

    Args:
        model: The model whose gradients to clip.
        clip_val: Clip by value (max absolute value).
        max_norm: Clip by norm (max norm).
        norm_type: Type of norm for clip-by-norm.

    Returns:
        Total norm if clip_by_norm was used, else None.
    """
    if clip_val is not None:
        paddle.nn.utils.clip_grad_value_(model.parameters(), clip_val)
        return None
    if max_norm is not None:
        return paddle.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type)
    return None
