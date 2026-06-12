"""Upgrade checkpoint format across Ocean versions."""


def upgrade_checkpoint(
    checkpoint: dict,
    from_version: str = "0.1.0",
    to_version: str = "0.2.0",
) -> dict:
    """Upgrade a checkpoint from one Ocean version to another.

    Args:
        checkpoint: Checkpoint dict to upgrade.
        from_version: Source version.
        to_version: Target version.

    Returns:
        Upgraded checkpoint.
    """
    ckpt = dict(checkpoint)
    ckpt.setdefault("paddle_ocean_version", to_version)
    ckpt.setdefault("epoch", 0)

    if "state_dict" not in ckpt and "model" in ckpt:
        ckpt["state_dict"] = ckpt.pop("model")
    if "optimizer_states" not in ckpt and "optimizer" in ckpt:
        ckpt["optimizer_states"] = [ckpt.pop("optimizer")]

    return ckpt
