"""HyperparametersMixin - save/load hyperparameters."""

import inspect
from copy import deepcopy
from typing import Any, Optional


class AttributeDict(dict):
    """A dict with attribute-style access."""
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"No attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class HyperparametersMixin:
    """Mixin that provides hparams property and save_hyperparameters method."""

    def __init__(self) -> None:
        self._hparams: Optional[AttributeDict] = None
        self._hparams_initial: Optional[dict[str, Any]] = None

    @property
    def hparams(self) -> AttributeDict:
        if self._hparams is None:
            self._hparams = AttributeDict()
        return self._hparams

    @hparams.setter
    def hparams(self, hp: dict[str, Any]) -> None:
        self._hparams = AttributeDict(hp)

    @property
    def hparams_initial(self) -> dict[str, Any]:
        if self._hparams_initial is None:
            return {}
        return deepcopy(self._hparams_initial)

    def save_hyperparameters(self, *args: Any, ignore: Optional[list[str]] = None, logger: bool = True) -> None:
        """Save hyperparameters. Supports three modes:

        1. No args: auto-capture __init__ parameters.
        2. Args as strings: capture specific __init__ parameters by name.
        3. Single dict/Namespace: use directly.
        """
        frame = inspect.currentframe()
        if frame is None:
            return
        try:
            parent_frame = frame.f_back
            if parent_frame is None:
                return
            if len(args) == 0:
                # Auto-capture all __init__ parameters
                init_sig = inspect.signature(self.__init__)
                init_params = list(init_sig.parameters.keys())[1:]  # skip self
                hp = {}
                for name in init_params:
                    if name in parent_frame.f_locals:
                        val = parent_frame.f_locals[name]
                        if not self._is_serializable(val):
                            val = str(type(val).__name__)
                        hp[name] = val
            elif len(args) == 1 and isinstance(args[0], dict):
                hp = dict(args[0])
            elif len(args) == 1 and hasattr(args[0], "__dict__"):
                hp = dict(vars(args[0]))
            else:
                hp = {}
                for arg in args:
                    if isinstance(arg, str) and arg in parent_frame.f_locals:
                        val = parent_frame.f_locals[arg]
                        if not self._is_serializable(val):
                            val = str(type(val).__name__)
                        hp[arg] = val

            if ignore:
                for key in ignore:
                    hp.pop(key, None)

            self._hparams = AttributeDict(hp)
            self._hparams_initial = deepcopy(hp)
        finally:
            del frame

    def _is_serializable(self, val: Any) -> bool:
        return isinstance(val, (int, float, str, bool, type(None), list, tuple, dict))
