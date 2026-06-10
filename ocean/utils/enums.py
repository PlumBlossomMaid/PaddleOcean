"""Enum utilities for paddleOcean."""

from enum import Enum as _Enum


class OceanEnum(_Enum):
    """Base enum class with string representation."""

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"
