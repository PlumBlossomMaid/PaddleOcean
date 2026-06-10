"""Accelerator base class for paddleOcean."""

from abc import ABC, abstractmethod
from typing import Any


class Accelerator(ABC):
    """Base accelerator class - manages device-level operations."""

    @abstractmethod
    def setup_device(self, device: Any) -> Any: ...

    def setup(self, trainer: Any) -> None: ...
    def teardown(self) -> None: ...

    @staticmethod
    @abstractmethod
    def parse_devices(devices: Any) -> Any: ...

    @staticmethod
    @abstractmethod
    def get_parallel_devices(devices: Any) -> list[Any]: ...

    @staticmethod
    @abstractmethod
    def auto_device_count() -> int: ...

    @staticmethod
    @abstractmethod
    def is_available() -> bool: ...

    def get_device_stats(self, device: Any) -> dict[str, Any]:
        return {}
