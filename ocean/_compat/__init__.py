"""_compat package: cross-version PaddlePaddle compatibility layer.

Provides version-agnostic wrappers for Paddle APIs that differ
between Paddle 2.4 and 3.3. Each missing API is implemented as
a pure-Paddle fallback (no C++ extensions) to ensure compatibility
with older versions.
"""

from ocean._compat.version import (
    Version,
    PADDLE_VERSION,
    V_2_4, V_2_5, V_2_6, V_3_0, V_3_1, V_3_2, V_3_3,
    version_gte, version_lt, api_available,
)
