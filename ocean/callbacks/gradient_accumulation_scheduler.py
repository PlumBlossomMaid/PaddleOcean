"""GradientAccumulationScheduler - changes gradient accumulation schedule."""

from typing import Any

from ocean.callbacks.callback import Callback
from ocean.utils import MisconfigurationException


class GradientAccumulationScheduler(Callback):
    """Change gradient accumulation factor during training.

    Args:
        scheduling: Dict mapping epoch -> accumulate_grad_batches value. Epoch
            keys must be non-negative ints and each accumulation factor a
            positive int. Between scheduled epochs the last set value persists.
    """

    def __init__(self, scheduling: dict[int, int]) -> None:
        if not scheduling:
            raise MisconfigurationException("Empty `scheduling` dict for GradientAccumulationScheduler.")
        for epoch, factor in scheduling.items():
            if not isinstance(epoch, int) or epoch < 0:
                raise MisconfigurationException(f"Epoch keys in `scheduling` must be non-negative ints, got {epoch!r}.")
            if not isinstance(factor, int) or factor < 1:
                raise MisconfigurationException(
                    f"Accumulation factors in `scheduling` must be ints >= 1, got {factor!r}."
                )
        self.scheduling = scheduling

    def on_train_epoch_start(self, trainer: Any, model: Any) -> None:
        epoch = trainer.current_epoch
        if epoch in self.scheduling:
            trainer.accumulate_grad_batches = self.scheduling[epoch]
