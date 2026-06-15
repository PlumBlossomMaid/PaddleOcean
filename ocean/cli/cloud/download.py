"""cloud download — download files from AI Studio."""

import os
from pathlib import Path
from typing import Optional

import click
import requests
from tqdm import tqdm

from ocean.cli.cloud import _config
from ocean.cli.cloud.auth import get_token


def _download_file(url: str, dest: str, token: str, desc: str = ""):
    """Download a single file with progress bar."""
    headers = {
        "Authorization": f"token {token}",
        "SDK-Version": "ocean-0.1",
    }
    resp = requests.get(url, headers=headers, stream=True, timeout=3600)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f:
        if total and desc:
            pbar = tqdm(total=total, unit="B", unit_scale=True, desc=desc)
        else:
            pbar = None
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            if pbar:
                pbar.update(len(chunk))
        if pbar:
            pbar.close()

    return dest


@click.command()
@click.argument("repo_id")
@click.argument("path_in_repo", required=False, default=None)
@click.option("--local-dir", default=".", help="Local directory to save files.")
@click.option("--repo-type", type=click.Choice(["model", "dataset"]), default="dataset")
@click.option("--revision", default="master", help="Branch name.")
@click.option("--token", default=None, help="AI Studio access token.")
@click.option("--include", multiple=True, help="Glob patterns to include.")
@click.option("--exclude", multiple=True, help="Glob patterns to exclude.")
def download(
    repo_id: str,
    path_in_repo: Optional[str],
    local_dir: str,
    repo_type: str,
    revision: str,
    token: Optional[str],
    include,
    exclude,
):
    """Download files from AI Studio.

    REPO_ID: Format ``username/repo-name``.

    Examples:

        ocean cloud download PlumBlossom/MyDataset

        ocean cloud download PlumBlossom/MyModel ./model.pdparams --repo-type model
    """
    token = token or get_token()
    _config.validate_repo_id(repo_id)

    dest = Path(local_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    git_host = os.getenv("STUDIO_GIT_HOST", _config.GIT_HOST)

    if path_in_repo:
        # Download single file
        url = f"{git_host}/{repo_id}/raw/branch/{revision}/{path_in_repo}"
        out = dest / Path(path_in_repo).name
        _download_file(url, str(out), token, desc=path_in_repo)
        click.echo(f"✅ Downloaded to {out}")
    else:
        # Download entire repo via API listing
        url = f"{git_host}/api/v1/repos/{repo_id}/contents?ref={revision}"
        headers = {
            "Authorization": f"token {token}",
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        items = resp.json()

        if isinstance(items, list):
            for item in items:
                if item["type"] == "file":
                    dl_path = dest / item["name"]
                    _download_file(
                        item["download_url"],
                        str(dl_path),
                        token,
                        desc=item["name"],
                    )
        elif isinstance(items, dict) and "download_url" in items:
            dl_path = dest / items["name"]
            _download_file(str(items["download_url"]), str(dl_path), token)

    click.echo(f"✅ Downloaded from {repo_id}")


# ── Public Python API ────────────────────────────────────────────────


def download_file(
    repo_id: str,
    path_in_repo: str,
    local_dir: str = ".",
    repo_type: str = "dataset",
    revision: str = "master",
    token: Optional[str] = None,
) -> str:
    """Download a single file from AI Studio.

    Args:
        repo_id: ``username/repo-name``.
        path_in_repo: File path within the repo.
        local_dir: Local directory to save to.
        repo_type: ``"dataset"`` or ``"model"``.
        revision: Branch name.
        token: AI Studio access token.

    Returns:
        Path to the downloaded file.

    Examples:
        >>> download_file("PlumBlossom/MyDataset", "data.zip")
    """
    token = token or get_token()
    _config.validate_repo_id(repo_id)
    git_host = os.getenv("STUDIO_GIT_HOST", _config.GIT_HOST)
    dest_dir = Path(local_dir).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"{git_host}/{repo_id}/raw/branch/{revision}/{path_in_repo}"
    out = dest_dir / Path(path_in_repo).name
    _download_file(url, str(out), token, desc=path_in_repo)
    return str(out)
