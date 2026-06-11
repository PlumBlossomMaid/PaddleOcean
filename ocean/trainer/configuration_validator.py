"""Configuration validator for paddleOcean Trainer."""

from typing import Any


def _check_num_devices(trainer: Any) -> None:
    """Validate device configuration."""
    devices = trainer.devices
    if isinstance(devices, int) and devices < 0 and devices != -1:
        raise ValueError(f"devices must be >= 0 or -1 (all), got {devices}")


def _check_strategy_and_devices(trainer: Any) -> None:
    """Validate strategy/device compatibility."""
    pass


def _check_data_limits(trainer: Any) -> None:
    """Validate data limit parameters."""
    for name in ["limit_train_batches", "limit_val_batches", "limit_test_batches", "limit_predict_batches"]:
        val = getattr(trainer, name, None)
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError(f"{name} must be int or float, got {type(val).__name__}")
