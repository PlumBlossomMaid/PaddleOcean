"""ocean.cloud — AI Studio cloud API.

Public API for uploading, downloading, and managing resources on
Baidu AI Studio. Works as both a Python library and CLI.

Examples:
    >>> from ocean.cloud import upload_file
    >>> upload_file("PlumBlossom/MyData", "./data.zip")

    >>> from ocean.cloud import download_file
    >>> download_file("PlumBlossom/MyData", "data.zip", local_dir="./data")

    >>> from ocean.cloud import list_files
    >>> list_files("PlumBlossom/MyData")
"""

from ocean.cli.cloud.auth import get_token, get_token_optional  # noqa: F401
from ocean.cli.cloud.delete import delete_file  # noqa: F401
from ocean.cli.cloud.download import download_file  # noqa: F401
from ocean.cli.cloud.list import list_files  # noqa: F401
from ocean.cli.cloud.upload import upload_file, upload_folder  # noqa: F401

__all__ = [
    "upload_file",
    "upload_folder",
    "download_file",
    "delete_file",
    "list_files",
    "get_token",
    "get_token_optional",
]
