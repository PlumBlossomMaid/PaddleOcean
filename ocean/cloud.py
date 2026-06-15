"""ocean.cloud — AI Studio cloud API.

Public API for uploading, downloading, and managing resources on
Baidu AI Studio. Works as both a Python library and CLI.

Examples:
    >>> from ocean.cloud import upload_file
    >>> upload_file("PlumBlossom/MyData", "./data.zip")
"""

from ocean.cli.cloud.auth import get_token  # noqa: F401
from ocean.cli.cloud.download import download_file  # noqa: F401
from ocean.cli.cloud.upload import upload_file, upload_folder  # noqa: F401
