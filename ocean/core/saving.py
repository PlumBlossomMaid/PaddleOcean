"""Checkpoint saving/loading utilities for Model and DataModule."""

from __future__ import annotations

import inspect
from typing import Any, Optional

import paddle

from ocean.utils.rank_zero import rank_zero_warn

CHECKPOINT_HYPER_PARAMS_KEY = "hyper_parameters"
#: Older checkpoints stored hyperparameters under this key.
CHECKPOINT_PAST_HYPER_PARAMS_KEYS = ("hparams",)


def _load_hparams_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Collect saved hyperparameters, newest key winning over legacy ones."""
    hparams: dict[str, Any] = {}
    for legacy_key in CHECKPOINT_PAST_HYPER_PARAMS_KEYS:
        hparams.update(checkpoint.get(legacy_key, {}) or {})
    hparams.update(checkpoint.get(CHECKPOINT_HYPER_PARAMS_KEY, {}) or {})
    return hparams


def _filter_init_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop entries ``cls.__init__`` cannot accept.

    A class that takes ``**kwargs`` accepts everything, so nothing is dropped.
    Otherwise a stale hyperparameter left in an old checkpoint would raise
    ``TypeError`` instead of simply being ignored.
    """
    try:
        spec = inspect.getfullargspec(cls.__init__)
    except TypeError:  # pragma: no cover - builtin/slot __init__
        return dict(kwargs)
    if spec.varkw:
        return dict(kwargs)
    accepted = set(spec.args[1:]) | set(spec.kwonlyargs)
    return {k: v for k, v in kwargs.items() if k in accepted}


def load_from_checkpoint(
    cls,
    checkpoint_path: str,
    map_location: Optional[str] = None,
    strict: bool = True,
    **kwargs: Any,
) -> Any:
    """Load a model (or datamodule) from a checkpoint file.

    The class is re-instantiated from the hyperparameters stored in the
    checkpoint, so it must have been saved with ``save_hyperparameters()`` for
    any ``__init__`` argument to be restored. ``**kwargs`` override the stored
    values.

    Args:
        cls: The class to instantiate.
        checkpoint_path: Path to the checkpoint file.
        map_location: Device to load tensors to.
        strict: Whether the checkpoint's keys must match the model's exactly.
        **kwargs: Values overriding the stored hyperparameters.

    Returns:
        An instance of ``cls`` with its state restored.
    """
    checkpoint = paddle.load(checkpoint_path)

    hparams = _load_hparams_from_checkpoint(checkpoint)
    hparams.update(kwargs)
    obj = cls(**_filter_init_kwargs(cls, hparams))

    state_dict = checkpoint.get("state_dict")
    if state_dict is not None and hasattr(obj, "set_state_dict"):
        missing, unexpected = obj.set_state_dict(state_dict)
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Error(s) in loading state_dict for {cls.__name__}:\n"
                f"\tMissing key(s) in state_dict: {sorted(missing)}\n"
                f"\tUnexpected key(s) in state_dict: {sorted(unexpected)}\n"
                "Pass `strict=False` to load anyway."
            )

    if map_location is not None:
        obj.to(map_location)

    if hasattr(obj, "on_load_checkpoint"):
        obj.on_load_checkpoint(checkpoint)

    return obj


def save_hparams_to_yaml(hparams: dict[str, Any], path: str) -> None:
    """Save hyperparameters to a YAML file.

    Says so when PyYAML is missing rather than returning as if it had written
    the file: hyperparameters vanishing without a word is exactly the kind of
    thing nobody notices until they need them.
    """
    try:
        import yaml
    except ImportError:
        rank_zero_warn(f"PyYAML is not installed, so the hyperparameters were not written to {path!r}.")
        return

    with open(path, "w") as f:
        yaml.dump(dict(hparams), f, default_flow_style=False)


def load_hparams_from_yaml(path: str) -> dict[str, Any]:
    """Load hyperparameters from a YAML file."""
    try:
        import yaml

        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return {}
