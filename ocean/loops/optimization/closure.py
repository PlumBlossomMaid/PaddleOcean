"""Closure classes for optimization steps."""

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

import paddle

T = TypeVar("T")


class OutputResult:
    """Result of a closure/step call."""

    def __init__(self, loss: Optional[paddle.Tensor] = None, outputs: Any = None) -> None:
        self.loss = loss
        self.outputs = outputs

    def asdict(self) -> dict[str, Any]:
        d = {}
        if self.loss is not None:
            d["loss"] = self.loss
        if self.outputs is not None:
            d["outputs"] = self.outputs
        return d


class AbstractClosure(ABC, Generic[T]):
    """Abstract base for optimization closures."""

    def __init__(self) -> None:
        self._result: Optional[T] = None

    def consume_result(self) -> T:
        result = self._result
        self._result = None
        if result is None:
            raise RuntimeError("No result available - closure may not have been called")
        return result

    @abstractmethod
    def closure(self, *args: Any, **kwargs: Any) -> T:
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._result = self.closure(*args, **kwargs)
        return self._result.loss if isinstance(self._result, OutputResult) else self._result
