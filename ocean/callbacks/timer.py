"""Timer callback - stops training after a time duration."""

import time
from datetime import timedelta
from typing import Any, Optional, Union

from ocean.callbacks.callback import Callback


class Timer(Callback):
    """Stop training after a time duration.

    Args:
        duration: Maximum training duration. String format 'DD:HH:MM:SS', timedelta, or dict.
        interval: Check interval ('step' or 'epoch').
        verbose: If True, print time elapsed/remaining messages.
    """

    def __init__(
        self,
        duration: Optional[Union[str, timedelta, dict[str, int]]] = None,
        interval: str = "step",
        verbose: bool = True,
    ) -> None:
        self._duration = self._parse_duration(duration)
        self.interval = interval
        self.verbose = verbose
        self._start_time: dict[str, float] = {}

    @property
    def duration(self) -> Optional[float]:
        return self._duration

    @staticmethod
    def _parse_duration(duration: Optional[Union[str, timedelta, dict[str, int]]]) -> Optional[float]:
        if duration is None:
            return None
        if isinstance(duration, timedelta):
            return duration.total_seconds()
        if isinstance(duration, dict):
            return timedelta(**duration).total_seconds()
        if isinstance(duration, str):
            parts = duration.split(":")
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            if len(parts) == 4:
                return int(parts[0]) * 86400 + int(parts[1]) * 3600 + int(parts[2]) * 60 + int(parts[3])
        raise ValueError(f"Invalid duration format: {duration}")

    def on_fit_start(self, trainer: Any, model: Any) -> None:
        self._start_time["fit"] = time.monotonic()

    def on_train_start(self, trainer: Any, model: Any) -> None:
        self._start_time["train"] = time.monotonic()

    def on_train_batch_end(self, trainer: Any, model: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self.interval == "step" and self._duration is not None:
            self._check_time_remaining(trainer)

    def on_train_epoch_end(self, trainer: Any, model: Any) -> None:
        if self.interval == "epoch" and self._duration is not None:
            self._check_time_remaining(trainer)

    def _check_time_remaining(self, trainer: Any) -> None:
        elapsed = time.monotonic() - self._start_time.get("fit", time.monotonic())
        if elapsed >= self._duration:
            trainer.should_stop = True
            if self.verbose:
                print(f"Timer: Reached duration limit ({self._duration}s), stopping training")

    def time_elapsed(self, stage: str = "train") -> float:
        return time.monotonic() - self._start_time.get(stage, time.monotonic())

    def time_remaining(self, stage: str = "train") -> Optional[float]:
        if self._duration is None:
            return None
        return max(0, self._duration - self.time_elapsed(stage))

    def state_dict(self) -> dict[str, Any]:
        return {"duration": self._duration, "start_time": dict(self._start_time)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._duration = state_dict.get("duration")
        self._start_time = state_dict.get("start_time", {})
