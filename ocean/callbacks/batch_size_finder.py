"""Batch size finder callback - finds max batch size that fits in memory."""

from typing import Any

from ocean.callbacks.callback import Callback


class BatchSizeFinder(Callback):
    """Find the maximum batch size that fits in memory.

    Args:
        mode: 'power' (double each trial) or 'binsearch'.
        steps_per_trial: Number of steps per trial.
        init_val: Initial batch size.
        max_trials: Maximum number of trials.
    """

    def __init__(
        self,
        mode: str = "power",
        steps_per_trial: int = 3,
        init_val: int = 2,
        max_trials: int = 25,
    ) -> None:
        self.mode = mode
        self.steps_per_trial = steps_per_trial
        self.init_val = init_val
        self.max_trials = max_trials
        self.optimal_batch_size: int = init_val

    def on_fit_start(self, trainer: Any, model: Any) -> None:
        self._original_limit = trainer.limit_train_batches
        trainer.limit_train_batches = self.steps_per_trial

    def on_train_epoch_end(self, trainer: Any, model: Any) -> None:
        # Simplified: scale batch size logic is handled by Tuner
        pass
