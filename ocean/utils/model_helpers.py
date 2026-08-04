"""Model helpers for ocean."""

from typing import Any


def _restricted_classmethod(func: Any) -> Any:
    """Decorator for classmethods that should raise an informative error."""
    import functools

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return classmethod(wrapper)


class _ModuleMode:
    """Captures every sublayer's ``training`` flag so it can be restored later.

    Prediction and evaluation force the model into eval mode. Restoring with a
    blanket ``train()`` would be wrong for a model that deliberately keeps parts
    of itself frozen in eval mode (a frozen BatchNorm backbone, say), so the
    per-sublayer flags are captured and put back exactly as they were.
    """

    def __init__(self) -> None:
        self.mode: dict[str, bool] = {}

    def capture(self, module: Any) -> None:
        self.mode.clear()
        for name, mod in module.named_sublayers(include_self=True):
            self.mode[name] = mod.training

    def restore(self, module: Any) -> None:
        for name, mod in module.named_sublayers(include_self=True):
            if name in self.mode:
                mod.training = self.mode[name]
