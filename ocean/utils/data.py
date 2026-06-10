"""Data utility functions."""

from typing import Any

import paddle


def move_data_to_device(batch: Any, device: Any) -> Any:
    """Move a batch of data to the specified device."""
    if isinstance(batch, paddle.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_data_to_device(b, device) for b in batch)
    if isinstance(batch, dict):
        return {k: move_data_to_device(v, device) for k, v in batch.items()}
    return batch


def apply_to_collection(data: Any, dtype: Any, function: Any, *args: Any, **kwargs: Any) -> Any:
    """Recursively apply a function to all elements of a given dtype in a nested collection."""
    if isinstance(data, dtype):
        return function(data, *args, **kwargs)
    if isinstance(data, dict):
        return {k: apply_to_collection(v, dtype, function, *args, **kwargs) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(apply_to_collection(v, dtype, function, *args, **kwargs) for v in data)
    return data
