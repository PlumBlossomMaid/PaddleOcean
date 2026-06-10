"""Core tensor operations compatibility layer.

Provides fallback implementations for tensor operations that may not exist
in older PaddlePaddle versions (< 3.0).
"""

from typing import Any, List, Optional, Sequence, Tuple, Union

import paddle

from ocean._compat.version import PADDLE_VERSION, V_2_5, V_3_0, version_gte


# ====================================================================
# Creation ops
# ====================================================================

def tensor(data: Any, dtype: Any = None, place: Any = None) -> paddle.Tensor:
    """Create a tensor (paddle.to_tensor wrapper for compatibility)."""
    return paddle.to_tensor(data, dtype=dtype, place=place)


def as_tensor(data: Any, dtype: Any = None, place: Any = None) -> paddle.Tensor:
    """Convert to tensor without copy when possible."""
    if isinstance(data, paddle.Tensor):
        if dtype is not None and data.dtype != paddle.dtype(dtype):
            return data.astype(dtype)
        return data
    return paddle.to_tensor(data, dtype=dtype, place=place)


# ====================================================================
# Indexing / scattering
# ====================================================================

def repeat_interleave(
    x: paddle.Tensor,
    repeats: Union[int, paddle.Tensor],
    axis: Optional[int] = None,
) -> paddle.Tensor:
    """Repeat elements of a tensor.

    Available natively in Paddle >= 2.5.
    Provides a pure-Paddle fallback for older versions.

    Reference: paddle.repeat_interleave (added in 2.5)
    """
    if version_gte("2.5") and hasattr(paddle, "repeat_interleave"):
        return paddle.repeat_interleave(x, repeats, axis=axis)

    # Pure-Paddle fallback
    if not isinstance(repeats, paddle.Tensor):
        repeats = paddle.to_tensor(repeats)

    if axis is None:
        x = x.flatten()
        axis = 0

    if repeats.numel() == 1:
        repeats = int(repeats)
        shape = list(x.shape)
        shape[axis] = 1
        x_exp = x.unsqueeze(axis + 1)
        expand_shape = list(x.shape)
        expand_shape.insert(axis + 1, repeats)
        x_exp = x_exp.expand(expand_shape)
        return x_exp.reshape([-1])

    # Per-element repeats
    indices = []
    for i in range(x.shape[axis]):
        r = int(repeats[i]) if repeats.numel() > 1 else int(repeats)
        indices.extend([i] * r)
    indices = paddle.to_tensor(indices, dtype="int64")
    return paddle.index_select(x, index=indices, axis=axis)


def index_add(
    x: paddle.Tensor,
    index: paddle.Tensor,
    axis: int,
    source: paddle.Tensor,
    alpha: float = 1.0,
) -> paddle.Tensor:
    """Add elements from source to x at positions given by index.

    Fallback for paddle.index_add (may not exist in very old versions).
    """
    if hasattr(paddle, "index_add"):
        return paddle.index_add(x, index, axis, source, alpha=alpha)

    # Fallback: one-hot scatter add
    if alpha != 1.0:
        source = source * alpha
    one_hot = paddle.nn.functional.one_hot(index, x.shape[axis])
    one_hot = one_hot.astype(x.dtype)
    if axis == 0:
        add_term = paddle.mm(one_hot.T, source)
    else:
        add_term = paddle.matmul(one_hot.T, source)
    return x + add_term


def scatter_along_axis(
    x: paddle.Tensor,
    index: paddle.Tensor,
    value: Any,
    axis: int = 0,
    reduce: str = "none",
) -> paddle.Tensor:
    """Scatter values along an axis (matching PyTorch's scatter_ semantics).

    Args:
        x: Source tensor.
        index: Index tensor.
        value: Values to scatter (tensor or scalar).
        axis: Axis along which to scatter.
        reduce: Reduction mode ('none' uses 'assign', 'add', 'multiply').
    """
    result = x.clone()
    paddle_reduce = "assign" if reduce == "none" else reduce
    if isinstance(value, paddle.Tensor) and value.ndim > 0:
        result = paddle.put_along_axis(result, index, value, axis=axis, reduce=paddle_reduce)
    else:
        value_t = paddle.full(index.shape, value, dtype=x.dtype)
        result = paddle.put_along_axis(result, index, value_t, axis=axis, reduce=paddle_reduce)
    return result


def scatter_nd(
    index: paddle.Tensor,
    updates: paddle.Tensor,
    shape: List[int],
) -> paddle.Tensor:
    """Scatter updates into a new tensor according to indices.

    Reference: paddle.scatter_nd
    """
    if hasattr(paddle, "scatter_nd"):
        return paddle.scatter_nd(index, updates, shape)
    raise NotImplementedError("scatter_nd requires Paddle >= 2.5")


def take_along_axis(
    x: paddle.Tensor,
    index: paddle.Tensor,
    axis: int = -1,
) -> paddle.Tensor:
    """Select elements from x according to index along axis.

    Reference: paddle.take_along_axis
    """
    if hasattr(paddle, "take_along_axis"):
        return paddle.take_along_axis(x, index, axis)
    # Fallback using gather
    return paddle.gather(x, index, axis=axis)


def put_along_axis(
    x: paddle.Tensor,
    index: paddle.Tensor,
    value: paddle.Tensor,
    axis: int = -1,
    reduce: str = "none",
) -> paddle.Tensor:
    """Put values into x according to index along axis.

    Reference: paddle.put_along_axis
    """
    if hasattr(paddle, "put_along_axis"):
        return paddle.put_along_axis(x, index, value, axis, reduce)
    result = x.clone()
    gathered = paddle.gather(x, index, axis=axis)
    result = paddle.scatter(result, index, value)
    return result


# ====================================================================
# Masking
# ====================================================================

def masked_fill(x: paddle.Tensor, mask: paddle.Tensor, value: Any) -> paddle.Tensor:
    """Fill elements with value where mask is True.

    Reference: paddle.Tensor.masked_fill_
    """
    if hasattr(paddle.Tensor, "masked_fill_"):
        return x.masked_fill_(mask, value)
    return paddle.where(mask, paddle.to_tensor(value, dtype=x.dtype), x)


def masked_select(x: paddle.Tensor, mask: paddle.Tensor) -> paddle.Tensor:
    """Select elements where mask is True.

    Reference: paddle.masked_select
    """
    if hasattr(paddle, "masked_select"):
        return paddle.masked_select(x, mask)
    return x.flatten()[mask.flatten()]


# ====================================================================
# Sorting / Searching
# ====================================================================

def sort(
    x: paddle.Tensor,
    axis: int = -1,
    descending: bool = False,
    stable: bool = False,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Sort elements along axis, return (sorted_values, indices).

    Reference: paddle.sort (returns only values), paddle.argsort (returns indices)
    Uses take_along_axis for compatibility with Paddle's gather semantics.
    """
    indices = paddle.argsort(x, axis=axis, descending=descending)
    sorted_x = paddle.take_along_axis(x, indices, axis=axis)
    return sorted_x, indices


def argsort(
    x: paddle.Tensor,
    axis: int = -1,
    descending: bool = False,
    stable: bool = False,
) -> paddle.Tensor:
    """Return indices that sort the tensor.

    Reference: paddle.argsort
    """
    if hasattr(paddle, "argsort"):
        if stable:
            return paddle.argsort(x, axis=axis, descending=descending)
        return paddle.argsort(x, axis=axis, descending=descending)
    return paddle.argsort(x, axis=axis, descending=descending)


# ====================================================================
# Unique / Nonzero
# ====================================================================

def unique(
    x: paddle.Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: Optional[int] = None,
) -> Tuple[paddle.Tensor, ...]:
    """Return unique elements.

    Reference: paddle.unique
    """
    if hasattr(paddle, "unique"):
        return paddle.unique(x, return_inverse=return_inverse, return_counts=return_counts, axis=axis)
    # Fallback for very old versions
    flat = x.flatten()
    sorted_t, _ = paddle.sort(flat)
    mask = sorted_t[1:] != sorted_t[:-1]
    mask = paddle.concat([paddle.to_tensor([True]), mask])
    uniq = sorted_t[mask]
    result = [uniq]
    if return_inverse:
        inverse = paddle.zeros_like(flat, dtype="int64")
        for i in range(uniq.numel()):
            inverse[flat == uniq[i]] = i
        result.append(inverse)
    if return_counts:
        counts = paddle.zeros_like(uniq, dtype="int64")
        for i in range(uniq.numel()):
            counts[i] = (flat == uniq[i]).sum()
        result.append(counts)
    return tuple(result)


def nonzero(x: paddle.Tensor, as_tuple: bool = False) -> Any:
    """Return indices of nonzero elements.

    Reference: paddle.nonzero
    """
    if hasattr(paddle, "nonzero"):
        result = paddle.nonzero(x)
    else:
        flat = x.flatten()
        indices = paddle.arange(flat.numel())
        result = indices[flat != 0].unsqueeze(1)
    if as_tuple:
        return tuple(result[:, i] for i in range(result.shape[1]))
    return result


# ====================================================================
# Math / Special
# ====================================================================

def logsumexp(x: paddle.Tensor, axis: Optional[int] = None, keepdim: bool = False) -> paddle.Tensor:
    """Logarithm of sum of exponentials.

    Reference: paddle.logsumexp
    """
    if hasattr(paddle, "logsumexp"):
        return paddle.logsumexp(x, axis=axis, keepdim=keepdim)
    if axis is None:
        x_max = x.max()
        return x_max + (x - x_max).exp().sum().log()
    x_max = x.max(axis=axis, keepdim=True)
    result = x_max + (x - x_max).exp().sum(axis=axis, keepdim=keepdim).log()
    return result


def lgamma(x: paddle.Tensor) -> paddle.Tensor:
    """Logarithm of gamma function.

    Reference: paddle.lgamma
    """
    if hasattr(paddle, "lgamma"):
        return paddle.lgamma(x)
    raise NotImplementedError("lgamma requires a Paddle version with implementation")


# ====================================================================
# NN compatibility
# ====================================================================

def pad(
    x: paddle.Tensor,
    pad: List[int],
    mode: str = "constant",
    value: float = 0.0,
) -> paddle.Tensor:
    """Pad tensor.

    Reference: paddle.nn.functional.pad
    """
    return paddle.nn.functional.pad(x, pad, mode=mode, value=value)


def one_hot(x: paddle.Tensor, num_classes: int) -> paddle.Tensor:
    """Convert integer labels to one-hot encoding.

    Reference: paddle.nn.functional.one_hot
    """
    return paddle.nn.functional.one_hot(x, num_classes)
