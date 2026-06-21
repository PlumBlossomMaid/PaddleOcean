"""cloud list — list files in AI Studio repos.

Usage:
    ocean cloud list user/repo
    ocean cloud list user/repo --repo-type model
    ocean cloud list user/repo path/to/subdir --repo-type dataset
"""

from typing import Optional

import click
import requests

from ocean.cli.cloud._config import GIT_HOST
from ocean.cli.cloud.auth import get_token
from ocean.cli.cloud.upload import _header_fill


def _fmt_size(size: int) -> str:
    """Format file size in human-readable form."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def list_files(
    repo_id: str,
    path: str = "",
    repo_type: str = "dataset",
    revision: str = "master",
    token: Optional[str] = None,
) -> list[dict]:
    """List files in an AI Studio repo.

    Args:
        repo_id: ``username/repo-name`` or ``user_id/repo_id``.
        path: Path within repo to list (default: root).
        repo_type: ``"dataset"`` or ``"model"``.
        revision: Git revision (branch/tag).
        token: AI Studio access token. Falls back to stored token.

    Returns:
        List of file info dicts with keys: name, type (file/dir), size, path.
    """
    token = token or get_token()
    path = path.strip("/")

    # Build URL
    if path:
        url = f"{GIT_HOST}/api/v1/repos/{repo_id}/contents/{path}"
    else:
        url = f"{GIT_HOST}/api/v1/repos/{repo_id}/contents"
    url += f"?ref={revision}"

    resp = requests.get(url, headers=_header_fill(token), timeout=30)
    resp.raise_for_status()
    items = resp.json()

    results = []
    for item in items:
        results.append({
            "name": item.get("name", ""),
            "type": item.get("type", "file"),
            "size": item.get("size", 0),
            "path": f"{path}/{item['name']}" if path else item["name"],
        })
    return results


@click.command()
@click.argument("repo_id")
@click.argument("path", required=False, default="")
@click.option("--repo-type", type=click.Choice(["model", "dataset"]), default="dataset")
@click.option("--revision", default="master", help="Branch/tag (default: master)")
@click.option("--token", default=None, help="AI Studio access token.")
def list(repo_id: str, path: str, repo_type: str, revision: str, token: Optional[str]):
    """List files in an AI Studio repo.

    REPO_ID: username/repo-name (e.g. user/MyDataset).

    Examples:

        ocean cloud list user/MyDataset

        ocean cloud list user/MyDataset path/to/subdir

        ocean cloud list user/MyModel --repo-type model
    """
    try:
        items = list_files(repo_id, path, repo_type, revision, token)
        if not items:
            click.echo("(empty)")
            return

        # Separate dirs and files
        dirs = [i for i in items if i["type"] == "dir"]
        files = [i for i in items if i["type"] != "dir"]

        for d in dirs:
            click.echo(f"  📁 {d['name']}/")
        for f in files:
            size_str = _fmt_size(f["size"]) if f["size"] > 0 else ""
            click.echo(f"  📄 {f['name']}  ({size_str})" if size_str else f"  📄 {f['name']}")

    except requests.HTTPError as e:
        click.echo(f"Error: {e}", err=True)
        if e.response is not None and e.response.status_code == 404:
            click.echo(f"  Repo '{repo_id}' not found or path '{path}' does not exist.", err=True)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
