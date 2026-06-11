"""Checkpoint saving/loading utilities for Model."""

from typing import Any, Optional

import paddle


def load_from_checkpoint(
    cls,
    checkpoint_path: str,
    map_location: Optional[str] = None,
    strict: bool = True,
    **kwargs: Any,
) -> Any:
    """Load a model from a checkpoint file.

    Args:
        cls: The Model class to instantiate.
        checkpoint_path: Path to the checkpoint file.
        map_location: Device to load tensors to.
        strict: Whether to strictly enforce state_dict keys.
        **kwargs: Additional kwargs to override hparams.

    Returns:
        An instance of cls with loaded state.
    """
    checkpoint = paddle.load(checkpoint_path)

    # Extract hparams from checkpoint
    hparams = checkpoint.get("hyper_parameters", {})

    # Override with kwargs
    if kwargs:
        hparams.update(kwargs)

    # Instantiate the model
    if hparams and hasattr(cls, "_hparams_initial"):
        model = cls(**hparams)
    else:
        model = cls()

    # Load state dict
    if "state_dict" in checkpoint:
        if strict:
            model.set_state_dict(checkpoint["state_dict"])
        else:
            model.set_dict(checkpoint["state_dict"])

    # Trigger load checkpoint hook
    model.on_load_checkpoint(checkpoint)

    return model


def save_hparams_to_yaml(hparams: dict[str, Any], path: str) -> None:
    """Save hyperparameters to a YAML file."""
    try:
        import yaml
        with open(path, "w") as f:
            yaml.dump(hparams, f, default_flow_style=False)
    except ImportError:
        pass


def load_hparams_from_yaml(path: str) -> dict[str, Any]:
    """Load hyperparameters from a YAML file."""
    try:
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return {}
