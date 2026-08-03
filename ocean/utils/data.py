"""Data utility functions."""

from typing import Any, Optional

import paddle


def sized_len(dataloader: Any) -> Optional[int]:
    """Return ``len(dataloader)``, or ``None`` when it has no usable length.

    Paddle raises :class:`ValueError` (not ``TypeError``) from
    ``DataLoader.__len__`` when the underlying dataset is an
    ``IterableDataset``, so catch that too — an unsized loader is a normal
    streaming setup, not an error.
    """
    try:
        return len(dataloader)
    except (TypeError, AttributeError, NotImplementedError, ValueError):
        return None


def has_len(dataloader: Any) -> bool:
    """Whether the dataloader exposes a usable length."""
    return sized_len(dataloader) is not None


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
