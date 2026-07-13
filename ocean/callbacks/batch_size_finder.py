"""Batch size finder callback - finds max batch size that fits in memory."""

from ocean.callbacks.callback import Callback


class BatchSizeFinder(Callback):
    """Find the maximum batch size that fits in memory.

    The actual search is performed by the Tuner (``trainer.scale_batch_size`` /
    ``trainer.tune``), which snapshots and restores model state so it does not
    corrupt the weights. This callback only records the search configuration and
    the resulting batch size.

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
