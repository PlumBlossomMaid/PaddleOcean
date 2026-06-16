"""ocelogger - Unified logger wrapper for paddleOcean.

Analogous to ocean's LitLogger.
Provides a consistent interface for all loggers.
"""

from typing import Any, Optional, Union

from ocean.loggers.base import Logger


class OceanLogger:
    """Unified logger that wraps multiple loggers.

    Delegates log_metrics/log_hyperparams to all registered loggers.

    Args:
        loggers: A single Logger or list of Loggers.
    """

    def __init__(self, loggers: Optional[Union[Logger, list[Logger]]] = None) -> None:
        if loggers is None:
            self._loggers: list[Logger] = []
        elif isinstance(loggers, Logger):
            self._loggers = [loggers]
        else:
            self._loggers = list(loggers)

    @property
    def loggers(self) -> list[Logger]:
        return self._loggers

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        for lg in self._loggers:
            try:
                lg.log_metrics(metrics, step)
            except Exception:
                pass

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        for lg in self._loggers:
            try:
                lg.log_hyperparams(params)
            except Exception:
                pass

    def save(self) -> None:
        for lg in self._loggers:
            try:
                lg.save()
            except Exception:
                pass

    def finalize(self, status: str = "success") -> None:
        for lg in self._loggers:
            try:
                lg.finalize(status)
            except Exception:
                pass

    def __getitem__(self, idx: int) -> Logger:
        return self._loggers[idx]


# Convenience alias
Ocelogger = OceanLogger
