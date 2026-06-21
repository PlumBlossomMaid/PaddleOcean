"""ocean cloud — AI Studio integration.

Usage (CLI):
    ocean cloud upload user/repo ./file.zip
    ocean cloud download user/repo
    ocean cloud login --token xyz

Usage (Python API):
    >>> from ocean.cloud import upload_file, download_file
    >>> upload_file("user/repo", "./data.zip", repo_type="dataset")
"""

# Public Python API
# CLI commands
import click

from ocean.cli.cloud import _config  # noqa: F401
from ocean.cli.cloud.auth import get_token
from ocean.cli.cloud.auth import login as _login_cli
from ocean.cli.cloud.auth import login as _login_fn
from ocean.cli.cloud.auth import logout as _logout_cli
from ocean.cli.cloud.auth import logout as _logout_fn
from ocean.cli.cloud.delete import delete as _delete_cli
from ocean.cli.cloud.delete import delete_file as _delete_fn
from ocean.cli.cloud.download import download as _download_cli
from ocean.cli.cloud.download import download_file
from ocean.cli.cloud.job import job
from ocean.cli.cloud.list import list as _list_cli
from ocean.cli.cloud.upload import upload as _upload_cli
from ocean.cli.cloud.upload import upload_file, upload_folder


@click.group()
def cloud():
    """AI Studio cloud operations.

    Manage datasets, models, and training jobs on Baidu AI Studio.
    """


cloud.add_command(_login_cli)
cloud.add_command(_logout_cli)
cloud.add_command(_upload_cli)
cloud.add_command(_download_cli)
cloud.add_command(_list_cli)
cloud.add_command(_delete_cli)
cloud.add_command(job)
