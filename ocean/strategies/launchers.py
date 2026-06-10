"""Launcher base classes for spawning sub-processes in distributed training."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class _Launcher(ABC):
    """Base class for launching distributed processes."""

    @abstractmethod
    def launch(self, function: Any, *args: Any, **kwargs: Any) -> Any: ...

    def kill(self, *args: Any, **kwargs: Any) -> None: ...


class _SingleProcessLauncher(_Launcher):
    """Launcher that runs in the current process (no spawning)."""

    def launch(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)
