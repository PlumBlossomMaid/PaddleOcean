"""Logger base class - abstract interface for all loggers."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class Logger(ABC):
    """Base class for loggers. All loggers must implement name, version, log_metrics, and log_hyperparams."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the experiment name."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the experiment version."""

    @property
    def root_dir(self) -> Optional[str]:
        """Return the root directory for logs."""
        return None

    @property
    def log_dir(self) -> Optional[str]:
        """Return the log directory."""
        return None

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Record metrics."""

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Record hyperparameters."""

    def save(self) -> None:
        """Save log data to disk."""

    def finalize(self, status: str) -> None:
        """Finalize the experiment (success/failed/aborted)."""
        self.save()
