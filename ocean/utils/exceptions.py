"""Exceptions for ocean."""


class MisconfigurationError(Exception):
    """Exception raised when the framework is misconfigured."""


class ClusterEnvironmentError(Exception):
    """Exception raised for cluster environment issues."""


# Backward compatibility aliases
MisconfigurationException = MisconfigurationError
ClusterEnvironmentException = ClusterEnvironmentError
