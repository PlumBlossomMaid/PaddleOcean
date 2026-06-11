"""ParallelStrategy - base for multi-device strategies."""

from abc import abstractmethod
from typing import Any

from ocean.strategies.strategy import Strategy


class ParallelStrategy(Strategy):
    """Base strategy for parallel/distributed execution."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.parallel_devices: list = []
        self._layer_sync = None

    @property
    def global_rank(self) -> int:
        return 0

    @property
    def local_rank(self) -> int:
        return 0

    @property
    def node_rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    @property
    def is_global_zero(self) -> bool:
        return self.global_rank == 0

    @abstractmethod
    def root_device(self) -> Any: ...

    def all_gather(self, tensor: Any, group: Any = None, sync_grads: bool = False) -> Any:
        return tensor

    def reduce_boolean_decision(self, decision: bool, all: bool = True) -> bool:
        return decision

    def block_backward_sync(self):
        import contextlib
        return contextlib.nullcontext()
