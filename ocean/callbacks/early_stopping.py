"""EarlyStopping callback - stops training when a monitored metric stops improving."""

import math
from typing import Any, Optional

from ocean.callbacks.callback import Callback
from ocean.utils import MisconfigurationException


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
        if mode not in self.mode_dict:
            raise MisconfigurationException(f"`mode` can be {', '.join(self.mode_dict)}, got {mode!r}")
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
        if self.check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)

    def on_train_epoch_end(self, trainer: Any, model: Any) -> None:
        if not self.check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)

    def _should_skip_check(self, trainer: Any) -> bool:
        """Skip early stopping during sanity checking or fast_dev_run."""
        return getattr(trainer, "sanity_checking", False) or bool(getattr(trainer, "fast_dev_run", False))

    def _run_early_stopping_check(self, trainer: Any) -> None:
        logs = trainer._log_metrics_on_epoch
        if self.monitor not in logs:
            if self.strict:
                raise RuntimeError(f"EarlyStopping: metric '{self.monitor}' not found in logs")
            return

        should_stop, reason = self._evaluate_stopping_criteria(logs[self.monitor])

        # Every rank has to reach the same answer. Deciding locally means some
        # ranks leave the loop while others wait at the next collective, which
        # does not stop training — it hangs it. ``all=False`` so one rank asking
        # to stop is enough, matching the reference.
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
        if not should_stop:
            return

        trainer.should_stop = True
        self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            print(reason)

    def _evaluate_stopping_criteria(self, current: float) -> tuple[bool, Optional[str]]:
        """Decide locally whether to stop, and say why.

        Split out from the check so the decision can be reduced across ranks in
        one place instead of each branch setting ``should_stop`` on its own.
        """
        if self.check_finite and not math.isfinite(current):
            return True, f"EarlyStopping: {self.monitor}={current} is not finite, stopping"

        if self.divergence_threshold is not None:
            diverged = (
                current >= self.divergence_threshold if self.mode == "min" else current <= self.divergence_threshold
            )
            if diverged:
                return True, f"EarlyStopping: {self.monitor}={current} diverged beyond {self.divergence_threshold}"

        if self.stopping_threshold is not None:
            reached = current <= self.stopping_threshold if self.mode == "min" else current >= self.stopping_threshold
            if reached:
                return True, f"EarlyStopping: {self.monitor}={current} reached stopping threshold"

        if self.mode == "min":
            improved = current < self.best_score - self.min_delta
        else:
            improved = current > self.best_score + self.min_delta

        if improved:
            self.best_score = current
            self.wait_count = 0
            return False, None

        self.wait_count += 1
        if self.wait_count >= self.patience:
            return True, f"EarlyStopping: {self.monitor} did not improve for {self.patience} checks"
        return False, None

    @property
    def state_key(self) -> str:
        return self._generate_state_key(monitor=self.monitor, mode=self.mode)

    def state_dict(self) -> dict[str, Any]:
        """Return state dict for checkpoint resume (ocean-compatible)."""
        return {
            "wait_count": self.wait_count,
            "stopped_epoch": self.stopped_epoch,
            "best_score": self.best_score,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dict from checkpoint (ocean-compatible)."""
        self.wait_count = state_dict.get("wait_count", 0)
        self.stopped_epoch = state_dict.get("stopped_epoch", 0)
        self.best_score = state_dict.get("best_score", self.best_score)
