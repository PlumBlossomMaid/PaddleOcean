"""cloud download — download files from AI Studio.

Supports regular and LFS files (model checkpoints, large datasets).

Usage:
    ocean cloud download user/repo path/to/file --repo-type model
"""

import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import click
import requests

from ocean.cli.cloud import _config
from ocean.cli.cloud.auth import get_token
from ocean.cli.cloud.upload import ColoredTqdm, _git_api, _header_fill


def _download_file(url: str, dest: str, token: str, desc: str = ""):
    """Download a single file with rainbow progress bar."""
    headers = {
        "Authorization": f"token {token}",
        "SDK-Version": "ocean-0.1",
    }
    resp = requests.get(url, headers=headers, stream=True, timeout=3600)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    dest_path = Path(dest)
    with open(dest_path, "wb") as f:
        with ColoredTqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=desc or dest_path.name,
            leave=False,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))
    return dest


def _lfs_download_real(repo_id: str, oid: str, file_size: int, token: str, dest: str, desc: str = "") -> None:
    """Download real LFS content from BOS via LFS batch API."""
    user_name, repo_name = repo_id.split("/")

    # Step 1: LFS batch API → get download URL
    resp = _git_api(
        "POST",
        f"/{user_name}/{repo_name}.git/info/lfs/objects/batch",
        token,
        data={
            "operation": "download",
            "objects": [{"oid": oid, "size": file_size}],
            "transfers": ["lfs-standalone-file", "basic"],
            "ref": {"name": "refs/heads/master"},
            "hash_algo": "sha256",
        },
        content_type="application/vnd.git-lfs+json",
    )

    if not resp.get("objects"):
        raise click.ClickException(f"{desc}: no LFS object found for {oid}")

    obj = resp["objects"][0]
    if "error" in obj:
        raise click.ClickException(f"{desc}: LFS error: {obj['error']}")

    actions = obj.get("actions", {})
    download_info = actions.get("download", {})
    download_href = download_info.get("href", "")
    if not download_href:
        raise click.ClickException(f"{desc}: no download URL available")

    # Step 2: Download from BOS
    click.echo(f"  ☁️  {desc} ({file_size / 1024 / 1024:.1f} MB)")
    resp = requests.get(download_href, stream=True, timeout=7200)
    resp.raise_for_status()

    with open(dest, "wb") as f:
        with ColoredTqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=desc,
            leave=False,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def _download_with_lfs_check(repo_id: str, file_info: dict, dest: str, token: str, revision: str = "master"):
    """Download a file, handling LFS objects properly."""
    name = file_info["name"]

    # First try the media API (handles LFS content transparently)
    user_name, repo_name = repo_id.split("/")
    git_host = os.getenv("STUDIO_GIT_HOST", _config.GIT_HOST)
    url = (
        f"{git_host}/api/v1/repos/{quote(user_name, safe='')}/{quote(repo_name, safe='')}/media/{quote(name, safe='')}"
    )
    if revision != "master":
        url += f"?ref={quote(revision, safe='')}"

    _download_file(url, dest, token, desc=name)

    # Check if the downloaded file is actually an LFS pointer
    dest_path = Path(dest)
    if dest_path.stat().st_size < 500:
        with open(dest_path, "r") as f:
            first_line = f.readline().strip()
        if first_line == "version https://git-lfs.github.com/spec/v1":
            # Parse OID and download real content
            content = dest_path.read_text()
            oid_match = re.search(r"oid sha256:([a-f0-9]{64})", content)
            size_match = re.search(r"size (\d+)", content)
            if oid_match and size_match:
                oid = oid_match.group(1)
                file_size = int(size_match.group(1))
                _lfs_download_real(repo_id, oid, file_size, token, dest, desc=name)
            # Remove the temporary pointer file if LFS download succeeded
            # (_lfs_download_real already wrote the real content to dest)


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
        # Download single file — get file info from API
        url = f"{git_host}/api/v1/repos/{repo_id}/contents/{path_in_repo}?ref={revision}"
        resp = requests.get(url, headers=_header_fill(token), timeout=30)
        resp.raise_for_status()
        file_info = resp.json()
        out = str(dest / Path(path_in_repo).name)
        _download_with_lfs_check(repo_id, file_info, out, token, revision)
        click.echo(f"✅ Downloaded to {out}")
    else:
        # Download all files via API listing
        url = f"{git_host}/api/v1/repos/{repo_id}/contents?ref={revision}"
        resp = requests.get(url, headers=_header_fill(token), timeout=30)
        resp.raise_for_status()
        items = resp.json()

        if isinstance(items, list):
            for item in items:
                if item["type"] == "file":
                    dl_path = str(dest / item["name"])
                    _download_with_lfs_check(repo_id, item, dl_path, token, revision)
        elif isinstance(items, dict):
            dl_path = str(dest / items["name"])
            _download_with_lfs_check(repo_id, items, dl_path, token, revision)

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

    Supports both regular files and Git LFS objects
    (e.g. model checkpoints uploaded via ``ocean cloud upload``).

    Args:
        repo_id: ``username/repo-name``.
        path_in_repo: File path within the repo.
        local_dir: Local directory to save to.
        repo_type: ``"dataset"`` or ``"model"``.
        revision: Branch name.
        token: AI Studio access token. Falls back to stored token.

    Returns:
        Path to the downloaded file.

    Examples:
        >>> download_file("PlumBlossom/MyModel", "rmvpe.pdparams", repo_type="model")
    """
    token = token or get_token()
    _config.validate_repo_id(repo_id)
    dest_dir = Path(local_dir).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Get file info (checks if LFS)
    git_host = os.getenv("STUDIO_GIT_HOST", _config.GIT_HOST)
    url = f"{git_host}/api/v1/repos/{repo_id}/contents/{path_in_repo}?ref={revision}"
    resp = requests.get(url, headers=_header_fill(token), timeout=30)
    resp.raise_for_status()
    file_info = resp.json()

    out = str(dest_dir / Path(path_in_repo).name)
    _download_with_lfs_check(repo_id, file_info, out, token, revision)
    return out
