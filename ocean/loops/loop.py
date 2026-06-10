"""Loop base class - provides state management for all loop types."""

from abc import ABC
from typing import Any, Optional


class _Loop(ABC):
    """Base class for all training/evaluation/prediction loops."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer
        self._restarting: bool = False
        self._loaded_from_state_dict: bool = False
        self._resuming_from_checkpoint: bool = False

    @property
    def restarting(self) -> bool:
        return self._restarting

    @restarting.setter
    def restarting(self, value: bool) -> None:
        self._restarting = value
        # Propagate to sub-loops
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, _Loop):
                attr.restarting = value

    @property
    def is_resuming(self) -> bool:
        return self._resuming_from_checkpoint

    def reset_restart_stage(self) -> None:
        """Reset the restart stage. Override in subclasses."""

    def on_save_checkpoint(self) -> dict[str, Any]:
        return {}

    def on_load_checkpoint(self, state_dict: dict[str, Any]) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        d = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, _Loop):
                d[attr_name] = attr.state_dict()
            elif hasattr(attr, "state_dict"):
                d[attr_name] = attr.state_dict()
        return d

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for attr_name, state in state_dict.items():
            attr = getattr(self, attr_name, None)
            if isinstance(attr, _Loop):
                attr.load_state_dict(state)
            elif hasattr(attr, "load_state_dict"):
                attr.load_state_dict(state)
        self._restarting = True
        self._resuming_from_checkpoint = True

    def on_iteration_done(self) -> None:
        self._restarting = False
