"""Loop utilities - helper functions for loops."""

from typing import Any


def _is_accumulating(trainer: Any) -> bool:
    """Check if the trainer is currently accumulating gradients."""
    return trainer.accumulate_grad_batches > 1
