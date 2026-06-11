"""GradientAccumulationScheduler - changes gradient accumulation schedule."""

from typing import Any

from ocean.callbacks.callback import Callback


class GradientAccumulationScheduler(Callback):
    """Change gradient accumulation factor during training.

    Args:
        scheduling: Dict mapping epoch -> accumulate_grad_batches value.
    """

    def __init__(self, scheduling: dict[int, int]) -> None:
        self.scheduling = scheduling

    def on_train_epoch_start(self, trainer: Any, model: Any) -> None:
        epoch = trainer.current_epoch
        if epoch in self.scheduling:
            trainer.accumulate_grad_batches = self.scheduling[epoch]
