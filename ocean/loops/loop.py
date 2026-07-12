"""Loop base class - provides state management for all loop types."""

from abc import ABC
from typing import Any


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
        for attr in self.__dict__.values():
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
        for name, attr in self.__dict__.items():
            if isinstance(attr, _Loop):
                d[name] = attr.state_dict()
            elif hasattr(attr, "state_dict"):
                d[name] = attr.state_dict()
        return d

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for name in state_dict:
            attr = getattr(self, name, None)
            if attr is None:
                continue
            if isinstance(attr, _Loop):
                attr.load_state_dict(state_dict[name])
            elif hasattr(attr, "load_state_dict"):
                attr.load_state_dict(state_dict[name])
        # Route through the property setter: it sets `self._restarting = True` AND
        # cascades the flag to every child _Loop (epoch_loop, automatic_optimization, ...),
        # so each nested loop's own restart branch actually engages.
        # A bare `self._restarting = True` would bypass the cascade and child loops would
        # silently run as fresh starts, resetting their batch_progress and re-processing
        # batches already covered by the checkpoint.
        self.restarting = True
        self._resuming_from_checkpoint = True

    def on_iteration_done(self) -> None:
        self._restarting = False
