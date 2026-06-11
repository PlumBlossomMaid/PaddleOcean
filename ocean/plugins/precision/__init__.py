"""Precision plugins package."""

from ocean.plugins.precision.amp import MixedPrecision
from ocean.plugins.precision.double import DoublePrecision
from ocean.plugins.precision.half import HalfPrecision
from ocean.plugins.precision.precision import Precision

PrecisionPlugin = Precision  # Alias for backward compatibility
