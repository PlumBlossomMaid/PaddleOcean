"""conftest for compat tests — centralized device blacklist."""


import paddle

# ------------------------------------------------------------------ #
# Device blacklist — single source of truth.
# ------------------------------------------------------------------ #
# Add devices whose PaddlePaddle plugin has incomplete kernel coverage
# here instead of scattering hardcoded names across individual tests.
#
# When a plugin is updated, simply remove the entry — tests re-enable
# automatically.
_DEVICE_BLACKLIST = {
    "iluvatar_gpu",  # 天数智芯 — contiguous(float64) kernel not registered
}

# Memoised device-type lookup.
_custom_device_type: str | None = None


def _get_device_type() -> str | None:
    global _custom_device_type
    if _custom_device_type is None:
        types = paddle.device.get_all_custom_device_type()
        _custom_device_type = types[0] if types else None
    return _custom_device_type


def in_device_blacklist() -> bool:
    """Check whether the current device is in the blacklist."""
    dev = _get_device_type()
    return dev in _DEVICE_BLACKLIST


def blacklist_skip_msg() -> str:
    """Descriptive skip message mentioning the device name."""
    dev = _get_device_type() or "unknown"
    return (
        f"Device '{dev}' is in the test blacklist "
        f"(see tests/compat/conftest.py _DEVICE_BLACKLIST)"
    )
