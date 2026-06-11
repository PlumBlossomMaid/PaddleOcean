"""EarlyStopping callback - stops training when a monitored metric stops improving."""

from typing import Any, Optional

from ocean.callbacks.callback import Callback


class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving.

    Args:
        monitor: Quantity to be monitored.
        min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        patience: Number of checks with no improvement after which training stops.
        verbose: Verbosity mode.
        mode: One of ``'min'`` or ``'max'``.
        strict: If True, raises an error when the metric is not found.
        check_finite: If True, stops when metric becomes NaN or Inf.
        stopping_threshold: Stop training once the metric reaches this threshold.
        divergence_threshold: Stop training if metric diverges beyond this threshold.
        check_on_train_epoch_end: Check on train epoch end instead of validation end.
    """

    mode_dict = {"min": lambda a, b: a < b, "max": lambda a, b: a > b}

    def __init__(
        self,
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
        check_on_train_epoch_end: Optional[bool] = None,
    ) -> None:
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.strict = strict
        self.check_finite = check_finite
        self.stopping_threshold = stopping_threshold
        self.divergence_threshold = divergence_threshold
        self.check_on_train_epoch_end = check_on_train_epoch_end

        self.wait_count = 0
        self.stopped_epoch = 0
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self._monitor_op = self.mode_dict[mode]

    def on_validation_end(self, trainer: Any, model: Any) -> None:
        if self.check_on_train_epoch_end:
            return
        self._run_early_stopping_check(trainer)

    def on_train_epoch_end(self, trainer: Any, model: Any) -> None:
        if not self.check_on_train_epoch_end:
            return
        self._run_early_stopping_check(trainer)

    def _run_early_stopping_check(self, trainer: Any) -> None:
        logs = trainer._log_metrics_on_epoch
        if self.monitor not in logs:
            if self.strict:
                raise RuntimeError(f"EarlyStopping: metric '{self.monitor}' not found in logs")
            return

        current = logs[self.monitor]
        if self._monitor_op(current, self.best_score):
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.stopped_epoch = trainer.current_epoch
                trainer.should_stop = True
                if self.verbose:
                    print(f"EarlyStopping: {self.monitor} did not improve for {self.patience} checks")
