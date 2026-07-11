"""HyperparametersMixin - save/load hyperparameters."""

import inspect
from copy import deepcopy
from typing import Any, Optional

import paddle

from ocean.utils import rank_zero_warn


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
            raise RuntimeError(
                "save_hyperparameters needs the CPython frame to introspect "
                "__init__ locals, which is unavailable on this interpreter."
            )
        try:
            parent_frame = frame.f_back
            if parent_frame is None:
                raise RuntimeError(
                    "save_hyperparameters could not locate the calling frame; call it directly inside __init__."
                )
            if len(args) == 0:
                # Auto-capture all __init__ parameters. inspect.signature on a
                # bound method already drops `self`, so don't slice again (doing
                # so would silently lose the first real parameter).
                init_sig = inspect.signature(self.__init__)
                init_params = list(init_sig.parameters.keys())
                hp = {}
                for name in init_params:
                    if name in parent_frame.f_locals:
                        hp[name] = self._normalize_hparam(name, parent_frame.f_locals[name])
            elif len(args) == 1 and isinstance(args[0], dict):
                hp = {k: self._normalize_hparam(k, v) for k, v in args[0].items()}
            elif len(args) == 1 and hasattr(args[0], "__dict__"):
                hp = {k: self._normalize_hparam(k, v) for k, v in vars(args[0]).items()}
            else:
                hp = {}
                for arg in args:
                    if isinstance(arg, str) and arg in parent_frame.f_locals:
                        hp[arg] = self._normalize_hparam(arg, parent_frame.f_locals[arg])

            if ignore:
                for key in ignore:
                    hp.pop(key, None)

            self._hparams = AttributeDict(hp)
            self._hparams_initial = deepcopy(hp)
        finally:
            del frame

    def _normalize_hparam(self, name: str, val: Any) -> Any:
        """Coerce a value into something serializable as a hyperparameter.

        Scalars and built-in containers survive; paddle/numpy tensors are pulled
        down to Python scalars (item/tolist) so their value is recorded rather
        than dropped. Anything else is logged as a type-name placeholder and a
        rank-zero warning warns that the value itself wasn't stored — instead of
        silently storing just the class name with no indication anything was lost.
        """
        if isinstance(val, (int, float, str, bool, type(None), list, tuple, dict)):
            return val
        # Paddle scalar -> python number; numpy scalar/list -> python
        if isinstance(val, paddle.Tensor):
            try:
                if val.numel() == 1:
                    return val.item()
                return val.tolist()
            except Exception:  # noqa: BLE001
                pass
        if "numpy" in str(type(val).__module__):
            try:
                if val.ndim == 0:
                    return val.item()
                return val.tolist()
            except Exception:  # noqa: BLE001
                pass
        placeholder = f"<{type(val).__name__}>"
        rank_zero_warn(
            f"Hyperparameter {name!r} of type {type(val).__name__} is not "
            f"serializable and was stored as the placeholder {placeholder!r} "
            f"rather than its value.",
        )
        return placeholder

    def _is_serializable(self, val: Any) -> bool:
        return isinstance(val, (int, float, str, bool, type(None), list, tuple, dict))
