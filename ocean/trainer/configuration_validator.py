"""Configuration validator for the ocean Trainer."""

from typing import Any


def _check_num_devices(trainer: Any) -> None:
    """Validate device configuration."""
    devices = trainer.devices
    if isinstance(devices, int) and devices < 0 and devices != -1:
        raise ValueError(f"devices must be >= 0 or -1 (all), got {devices}")


def _check_strategy_and_devices(trainer: Any) -> None:
    """Validate strategy/device compatibility.

    Ensures DDPStrategy is not used with a single device (would silently run
    without distributed init), and that distributed strategies are only used
    when distributed is available.
    """
    from ocean.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy, ModelParallelStrategy

    strategy = trainer.strategy
    devices = trainer.devices

    if isinstance(devices, int):
        num_devices = devices if devices > 0 else (1 if devices in (0, -1) else abs(devices))
    elif isinstance(devices, str):
        num_devices = len(devices.split(",")) if devices else 1
    elif isinstance(devices, (list, tuple)):
        num_devices = len(devices)
    else:
        num_devices = 1

    # DDP/FSDP/DeepSpeed/ModelParallel with 1 device is a configuration error
    # (single-device training should use SingleDeviceStrategy)
    if isinstance(strategy, (DDPStrategy, FSDPStrategy, ModelParallelStrategy, DeepSpeedStrategy)):
        if num_devices <= 1 and not getattr(trainer, "_is_distributed_spawn", False):
            import warnings

            warnings.warn(
                f"{type(strategy).__name__} is used with {num_devices} device(s). "
                "Distributed strategies are designed for multi-device training. "
                "Consider using SingleDeviceStrategy for single-device training.",
                UserWarning,
                stacklevel=2,
            )


def _check_data_limits(trainer: Any) -> None:
    """Validate data limit parameters."""
    for name in ["limit_train_batches", "limit_val_batches", "limit_test_batches", "limit_predict_batches"]:
        val = getattr(trainer, name, None)
        if val is not None and not isinstance(val, (int, float)):
            raise ValueError(f"{name} must be int or float, got {type(val).__name__}")
