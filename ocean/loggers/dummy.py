"""DummyLogger - a no-op logger used to suppress logging for a run."""

from __future__ import annotations

from typing import Any, Callable, Optional

from ocean.loggers.base import Logger
from ocean.utils.rank_zero import _DummyExperiment


class DummyLogger(Logger):
    """No-op logger.

    Used to disable a user's logger for a feature (e.g. ``fast_dev_run``) while
    keeping user code that touches ``trainer.logger`` working — every attribute
    access resolves to a no-op instead of raising ``AttributeError``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._experiment = _DummyExperiment()

    @property
    def experiment(self) -> _DummyExperiment:
        return self._experiment

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        pass

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        pass

    @property
    def name(self) -> str:
        return ""

    @property
    def version(self) -> str:
        return ""

    def __getitem__(self, idx: int) -> "DummyLogger":
        # Enables ``self.logger[0].experiment.add_image(...)``.
        return self

    def __getattr__(self, name: str) -> Callable:
        """Resolve any other attribute to a no-op callable."""

        def method(*args: Any, **kwargs: Any) -> None:
            return None

        return method
