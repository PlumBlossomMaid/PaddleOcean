"""Cluster environment base for distributed training."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class ClusterEnvironment(ABC):
    """Base class for cluster environment detection."""

    @abstractmethod
    def creates_children(self) -> bool: ...

    @abstractmethod
    def master_address(self) -> str: ...

    @abstractmethod
    def master_port(self) -> int: ...

    @abstractmethod
    def world_size(self) -> int: ...

    @abstractmethod
    def global_rank(self) -> int: ...

    @abstractmethod
    def local_rank(self) -> int: ...

    @abstractmethod
    def node_rank(self) -> int: ...

    def teardown(self) -> None: ...

    def is_shared_filesystem(self) -> bool:
        return True
