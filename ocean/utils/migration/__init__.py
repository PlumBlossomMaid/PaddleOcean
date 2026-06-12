"""Checkpoint migration utilities - upgrade checkpoints between paddleOcean versions."""

from typing import Any, Optional

import paddle


def migrate_checkpoint(
    checkpoint: dict,
    from_version: str = "0.1.0",
    to_version: str = "0.2.0",
) -> dict:
    """Migrate a checkpoint from one version to another.

    Args:
        checkpoint: The checkpoint dictionary to migrate.
        from_version: Source version.
        to_version: Target version.

    Returns:
        Migrated checkpoint.
    """
    ckpt = dict(checkpoint)

    # Handle old key naming conventions
    if "state_dict" not in ckpt and "model_state_dict" in ckpt:
        ckpt["state_dict"] = ckpt.pop("model_state_dict")

    if "optimizer_states" not in ckpt and "optimizer_state_dict" in ckpt:
        ckpt["optimizer_states"] = [ckpt.pop("optimizer_state_dict")]

    # Ensure metadata
    ckpt.setdefault("paddle_ocean_version", to_version)
    ckpt.setdefault("epoch", 0)

    return ckpt
