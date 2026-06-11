"""PaddlePaddle version detection and compatibility utilities.

Automatically detects the installed Paddle version and provides
- Version comparison helpers
- Decorators for version-gated APIs
- Structured version info
"""

import re

import paddle


class Version:
    """Semantic version representation for PaddlePaddle."""

    def __init__(self, major: int, minor: int, patch: int = 0) -> None:
        self.major = major
        self.minor = minor
        self.patch = patch

    @classmethod
    def parse(cls, version_str: str) -> "Version":
        """Parse a version string like '2.5.2' or '3.3.1'."""
        match = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", version_str)
        if not match:
            raise ValueError(f"Invalid version string: {version_str}")
        major, minor, patch = match.groups()
        return cls(int(major), int(minor), int(patch or 0))

    @classmethod
    def current(cls) -> "Version":
        """Get the current PaddlePaddle version."""
        return cls.parse(paddle.__version__)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __lt__(self, other: "Version") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __le__(self, other: "Version") -> bool:
        return (self.major, self.minor, self.patch) <= (other.major, other.minor, other.patch)

    def __gt__(self, other: "Version") -> bool:
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)

    def __ge__(self, other: "Version") -> bool:
        return (self.major, self.minor, self.patch) >= (other.major, other.minor, other.patch)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __repr__(self) -> str:
        return f"Version({self.major}, {self.minor}, {self.patch})"


# Current version
PADDLE_VERSION: Version = Version.current()

# Common version thresholds as constants
V_2_4 = Version(2, 4)
V_2_5 = Version(2, 5)
V_2_6 = Version(2, 6)
V_3_0 = Version(3, 0)
V_3_1 = Version(3, 1)
V_3_2 = Version(3, 2)
V_3_3 = Version(3, 3)


def version_gte(target: str) -> bool:
    """Check if current Paddle version >= target.

    Args:
        target: Version string like '2.5.0'.
    """
    return PADDLE_VERSION >= Version.parse(target)


def version_lt(target: str) -> bool:
    """Check if current Paddle version < target."""
    return PADDLE_VERSION < Version.parse(target)


def api_available(api_path: str) -> bool:
    """Check if a Paddle API is available in the current version.

    Args:
        api_path: Dot-separated path like 'paddle.repeat_interleave'

    Returns:
        True if the API exists.
    """
    parts = api_path.split(".")
    obj = __import__("paddle")
    for part in parts[1:]:
        obj = getattr(obj, part, None)
        if obj is None:
            return False
    return True
