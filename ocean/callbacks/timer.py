"""Timer callback - stop training after a time duration.

Tracks elapsed training time and stops training when a time budget is hit.

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
from ocean.trainer.states import RunningStage


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

        # Per-stage wall-clock windows. Training time is measured as one
        # contiguous window from ``on_train_start`` to ``on_train_end`` (matching
        # the reference); ``_offset`` carries elapsed training time across a
        # checkpoint resume so the budget survives restarts.
        self._start_time: dict[RunningStage, Optional[float]] = dict.fromkeys(RunningStage)
        self._end_time: dict[RunningStage, Optional[float]] = dict.fromkeys(RunningStage)
        self._offset: float = 0.0

    @property
    def duration(self) -> Optional[float]:
        return self._duration

    def start_time(self, stage: Union[str, RunningStage] = RunningStage.TRAINING) -> Optional[float]:
        """Return the monotonic start time recorded for ``stage`` (or None)."""
        return self._start_time[RunningStage(stage)]

    def end_time(self, stage: Union[str, RunningStage] = RunningStage.TRAINING) -> Optional[float]:
        """Return the monotonic end time recorded for ``stage`` (or None)."""
        return self._end_time[RunningStage(stage)]

    def time_elapsed(self, stage: Union[str, RunningStage] = RunningStage.TRAINING) -> float:
        """Seconds elapsed for ``stage``.

        For training this includes the resume ``_offset``. A still-running stage
        (start recorded, no end yet) is measured up to ``time.monotonic()``.
        """
        stage = RunningStage(stage)
        start = self._start_time[stage]
        end = self._end_time[stage]
        offset = self._offset if stage == RunningStage.TRAINING else 0.0
        if start is None:
            return offset
        if end is None:
            return time.monotonic() - start + offset
        return end - start + offset

    def time_remaining(self, stage: Union[str, RunningStage] = RunningStage.TRAINING) -> Optional[float]:
        """Seconds remaining against ``duration`` for ``stage`` (None if no budget)."""
        if self._duration is None:
            return None
        return self._duration - self.time_elapsed(stage)

    def _check_time_remaining(self, trainer: Any) -> None:
        if self._duration is None:
            return
        should_stop = self.time_elapsed() >= self._duration
        # Clocks drift, so ranks cross the deadline at slightly different times.
        # Deciding locally means some ranks leave the loop while others wait at
        # the next collective, which hangs the run instead of ending it; rank 0's
        # answer is the one that counts.
        should_stop = trainer.strategy.broadcast(should_stop)
        if should_stop:
            trainer.should_stop = True
            if self._verbose:
                print(f"Timer: training time limit of {self._duration:.0f}s reached, stopping.")

    def on_train_start(self, trainer: Any, model: Any) -> None:
        self._start_time[RunningStage.TRAINING] = time.monotonic()

    def on_train_end(self, trainer: Any, model: Any) -> None:
        self._end_time[RunningStage.TRAINING] = time.monotonic()

    def on_validation_start(self, trainer: Any, model: Any) -> None:
        self._start_time[RunningStage.VALIDATING] = time.monotonic()

    def on_validation_end(self, trainer: Any, model: Any) -> None:
        self._end_time[RunningStage.VALIDATING] = time.monotonic()

    def on_test_start(self, trainer: Any, model: Any) -> None:
        self._start_time[RunningStage.TESTING] = time.monotonic()

    def on_test_end(self, trainer: Any, model: Any) -> None:
        self._end_time[RunningStage.TESTING] = time.monotonic()

    def on_fit_start(self, trainer: Any, model: Any) -> None:
        # Check right after (a possibly resumed) state is in place, regardless of
        # interval, so a run whose budget is already spent stops immediately.
        if self._duration is None:
            return
        if self._verbose:
            remaining = self.time_remaining()
            if remaining is not None and remaining > 0:
                print(f"Timer: training will be interrupted after {self._duration:.0f} seconds")
        self._check_time_remaining(trainer)

    def on_train_batch_end(self, trainer: Any, model: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self._interval == "step":
            self._check_time_remaining(trainer)

    def on_train_epoch_end(self, trainer: Any, model: Any) -> None:
        if self._interval == "epoch":
            self._check_time_remaining(trainer)

    def state_dict(self) -> dict[str, Any]:
        return {"time_elapsed": {stage.name: self.time_elapsed(stage) for stage in RunningStage}}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        time_elapsed = state_dict.get("time_elapsed", {})
        self._offset = time_elapsed.get(RunningStage.TRAINING.name, 0.0)

    def __repr__(self) -> str:
        remaining = self.time_remaining()
        if remaining is not None:
            return f"Timer(duration={self._duration:.0f}s, remaining={remaining:.0f}s)"
        return "Timer(duration=None)"
