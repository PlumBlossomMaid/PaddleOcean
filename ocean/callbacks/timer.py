"""Timer callback - stop training after a time duration.

Fully aligned with paddleOcean's Timer callback (non-TPU, non-XLA parts).

Usage:
    from ocean.callbacks import Timer
    trainer = ocean.Trainer(
        max_time="00:12:00:00",  # 12 days
        callbacks=[Timer(...)]
    )
    # Or use the string "00:12:00:00" directly in max_time param.
"""

import time
from datetime import timedelta
from typing import Any, Optional, Union

from ocean.callbacks.callback import Callback


def _parse_duration(duration: Union[str, timedelta, dict, None]) -> Optional[float]:
    """Parse duration into total seconds.

    Supports:
        - str: "DD:HH:MM:SS" or "HH:MM:SS"
        - timedelta: datetime.timedelta object
        - dict: {"days": 1, "hours": 2, "minutes": 30}
        - None: returns None
        - float/int: treated as seconds
    """
    if duration is None:
        return None
    if isinstance(duration, (int, float)):
        return float(duration)
    if isinstance(duration, timedelta):
        return duration.total_seconds()
    if isinstance(duration, dict):
        return timedelta(**duration).total_seconds()
    if isinstance(duration, str):
        duration = duration.strip()
        parts = duration.split(":")
        if len(parts) == 4:  # DD:HH:MM:SS
            return timedelta(
                days=int(parts[0]),
                hours=int(parts[1]),
                minutes=int(parts[2]),
                seconds=int(parts[3]),
            ).total_seconds()
        elif len(parts) == 3:  # HH:MM:SS
            return timedelta(
                hours=int(parts[0]),
                minutes=int(parts[1]),
                seconds=int(parts[2]),
            ).total_seconds()
        elif len(parts) == 2:  # MM:SS
            return timedelta(
                minutes=int(parts[0]),
                seconds=int(parts[1]),
            ).total_seconds()
        elif len(parts) == 1:  # raw seconds
            return float(duration)
    raise ValueError(f"Invalid duration format: {duration}")


class Timer(Callback):
    """Stop training after a given time duration.

    Args:
        duration: Maximum training time. Can be a string ("HH:MM:SS" or "DD:HH:MM:SS"),
            a timedelta object, a dict, or a float/int (seconds).
        interval: Check time at 'step' or 'epoch' intervals. Default: "step".
        verbose: If True, log time remaining. Default: True.

    Example:
        >>> Timer(duration="00:12:00:00")  # 12 days
        >>> Timer(duration=timedelta(hours=3))  # 3 hours
        >>> Timer(duration={"minutes": 30})  # 30 minutes
    """

    def __init__(
        self,
        duration: Union[str, timedelta, dict, float, None] = None,
        interval: str = "step",
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self._duration = _parse_duration(duration)
        self._interval = interval
        self._verbose = verbose

        # Internal state
        self._start_time: Optional[float] = None
        self._training_start_time: Optional[float] = None
        self._total_train_time: float = 0.0
        self._epoch_time_start: Optional[float] = None

    @property
    def start_time(self) -> Optional[float]:
        return self._start_time

    @property
    def duration(self) -> Optional[float]:
        return self._duration

    @property
    def total_time(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def time_remaining(self) -> Optional[float]:
        if self._duration is None:
            return None
        elapsed = self.total_time - self._total_train_time
        return max(0.0, self._duration - elapsed)

    @property
    def on_training_start_time(self) -> Optional[float]:
        return self._training_start_time

    @property
    def training_time(self) -> float:
        return self._total_train_time

    def _check_time_remaining(self, trainer: Any) -> None:
        if self._duration is None:
            return
        remaining = self.time_remaining
        if remaining is not None and remaining <= 0:
            trainer.should_stop = True

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        """Mark start of training phase (ocean-compatible)."""
        self._training_start_time = time.time()
        self._epoch_time_start = time.time()

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self._epoch_time_start = time.time()

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        """Accumulate training time (ocean-compatible)."""
        if self._training_start_time is not None:
            self._total_train_time += time.time() - self._training_start_time

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        """Reset training timer after validation (ocean-compatible)."""
        self._training_start_time = time.time()

    def on_test_end(self, trainer: Any, pl_module: Any) -> None:
        """Reset training timer after testing (ocean-compatible)."""
        self._training_start_time = time.time()

    # ── Existing hooks ──

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        if self._start_time is None:
            self._start_time = time.time()
            self._training_start_time = 0.0

        # On checkpoint resume, adjust for already elapsed time
        if self._total_train_time > 0:
            self._start_time = time.time() - self._total_train_time

        if self._duration is None or not self._verbose:
            return
        remaining = self.time_remaining
        if remaining is not None and remaining > 0:
            print(f"Timer: Training will be interrupted after {self._duration:.0f} seconds")
        elif remaining is not None:
            print("Timer: Time limit reached during checkpoint load")
            trainer.should_stop = True

    def on_train_batch_end(self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self._interval == "step":
            self._check_time_remaining(trainer)

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if self._interval == "epoch":
            self._check_time_remaining(trainer)

    def state_dict(self) -> dict[str, Any]:
        return {
            "_start_time": self._start_time,
            "_total_train_time": self._total_train_time,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._start_time = state_dict.get("_start_time")
        self._total_train_time = state_dict.get("_total_train_time", 0.0)

    def __repr__(self) -> str:
        remaining = self.time_remaining
        if remaining is not None:
            return f"Timer(duration={self._duration:.0f}s, remaining={remaining:.0f}s)"
        return "Timer(duration=None)"
