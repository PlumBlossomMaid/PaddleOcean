"""cloud download — download files from AI Studio.

Supports regular and LFS files (model checkpoints, large datasets).
Uses ``git/trees?recursive=true`` API for listing all files,
and the ``media`` API for downloading each file.

Usage:
    ocean cloud download user/repo path/to/file --repo-type model
"""

import json
import os
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from json import JSONDecodeError
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import click
import requests

from ocean.cli.cloud import _config
from ocean.cli.cloud.auth import get_token_optional
from ocean.cli.cloud.list import list_files
from ocean.cli.cloud.upload import ColoredTqdm, _git_api, _header_fill

# ── Download manifest (persistent cache index) ──────────────────────
# Mirrors official SDK's ModelFileSystemCache but simpler.
# Stores {relative_path: git_blob_sha} so re-runs skip already-downloaded files.


_MANIFEST_LOCK = threading.Lock()
_MANIFEST_NAME = "_ocean_download_cache.json"


def _load_manifest(dest_dir: Path) -> dict:
    """Load download manifest from dest_dir."""
    path = dest_dir / _MANIFEST_NAME
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_manifest(dest_dir: Path, manifest: dict):
    """Atomically write manifest to dest_dir."""
    path = dest_dir / _MANIFEST_NAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))


def _is_cached(dest_dir: Path, file_path: str, expected_sha: str) -> bool:
    """Check if a file is fully downloaded and matches the expected git blob SHA."""
    local_path = dest_dir / file_path
    if not local_path.exists():
        return False
    with _MANIFEST_LOCK:
        manifest = _load_manifest(dest_dir)
    cached_sha = manifest.get(file_path)
    if cached_sha == expected_sha:
        # Quick size sanity check — empty or truncated file
        if local_path.stat().st_size > 0:
            return True
    return False


def _mark_cached(dest_dir: Path, file_path: str, sha: str):
    """Record file as successfully downloaded."""
    local_path = dest_dir / file_path
    if not local_path.exists():
        return
    with _MANIFEST_LOCK:
        manifest = _load_manifest(dest_dir)
        manifest[file_path] = sha
        _save_manifest(dest_dir, manifest)


def _auth_hint():
    return (
        "This repository requires authentication.\n"
        "  Run 'ocean cloud login --token YOUR_TOKEN' first, "
        "or set AISTUDIO_ACCESS_TOKEN environment variable."
    )


def _resolve_repo_paths(files: list[dict], root_items: list[str]) -> list[str]:
    """Resolve correct repo-relative paths from git/trees entries.

    AI Studio Gitea may return full server-side paths (e.g.
    ``code/home/aistudio/.../repo/file.py``) instead of repo-relative
    paths.  We use the Contents API (``list``) as a trusted reference:
    if every git/trees path ends with a known root-level entry name, we
    can correctly determine where the repo structure starts.

    Args:
        files: List of file info dicts from ``_list_all_files``.
        root_items: Known root-level entry names from the Contents API.

    Returns:
        List of corrected repo-relative paths in the same order as
        ``files``.
    """
    if not files or not root_items:
        # No reference to correct against — return paths as-is
        return [f["path"] for f in files]

    # Build a set of root-level names for fast lookup.
    # Only consider entries that actually appear in the git/trees paths.
    root_set = set(root_items)

    resolved = []
    for entry in files:
        path = entry["path"]
        parts = path.split("/")
        # Walk from the end of the path backwards — the first component
        # that matches a root entry name is the start of the repo
        # structure.
        matched = -1
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] in root_set:
                matched = i
                break
        if matched >= 0:
            resolved.append("/".join(parts[matched:]))
        else:
            # Fallback: keep the original path
            resolved.append(path)
    return resolved


def _list_all_files(repo_id: str, revision: str, token: str | None):
    """List all files in the repo using git/trees?recursive=true API.

    Returns list of dicts with keys: path, type, size, sha.
    """
    git_host = os.getenv("STUDIO_GIT_HOST", _config.GIT_HOST)
    headers = _header_fill(token)
    user_name, repo_name = repo_id.split("/")
    url_encoded_repo = f"{quote(user_name, safe='')}/{quote(repo_name, safe='')}"

    # Get the commit SHA for the revision (branch/tag)
    tag_url = f"{git_host}/api/v1/repos/{url_encoded_repo}/tags/{quote(revision, safe='')}"
    tag_resp = requests.get(tag_url, headers=headers, timeout=30)
    if tag_resp.status_code in (401, 403):
        raise click.ClickException(_auth_hint())
    if tag_resp.status_code == 200:
        revision_sha = tag_resp.json()["commit"]["sha"]
    else:
        # Fallback: use revision as-is (could be a commit SHA)
        revision_sha = revision

    # Paginate through git/trees
    page = 1
    per_page = 1000
    all_files = []

    while True:
        tree_url = (
            f"{git_host}/api/v1/repos/{url_encoded_repo}/git/trees/{quote(revision_sha, safe='')}"
            f"?recursive=true&page={page}&per_page={per_page}"
        )
        resp = requests.get(tree_url, headers=headers, timeout=30)
        if resp.status_code in (401, 403):
            raise click.ClickException(_auth_hint())
        if resp.status_code not in (200, 201):
            raise click.ClickException(f"Failed to list repo: {resp.status_code} {resp.text[:200]}")

        d = resp.json()
        tree = d.get("tree", [])
        for entry in tree:
            if entry.get("type") == "blob":
                path = entry["path"]
                # Skip .gitignore, .gitattributes
                if path in (".gitignore", ".gitattributes"):
                    continue
                all_files.append(entry)

        if not d.get("truncated", False):
            break
        page += 1

    return all_files


def _download_file(url: str, dest: str, token: str | None, desc: str = "", file_size: int = 0):
    """Download a single file with rainbow progress bar."""
    headers = {"SDK-Version": "ocean-0.1"}
    if token:
        headers["Authorization"] = f"token {token}"
    resp = requests.get(url, headers=headers, stream=True, timeout=3600)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0)) or file_size
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

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


def _lfs_download_real(repo_id: str, oid: str, file_size: int, token: str | None, dest: str, desc: str = "") -> bool:
    """Download real LFS content from BOS via LFS batch API.
    Returns True on success, False if the LFS object is missing."""
    user_name, repo_name = repo_id.split("/")

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
        click.echo(f"  ⚠️  {desc}: LFS object not found")
        return False

    obj = resp["objects"][0]
    if "error" in obj:
        code = obj["error"].get("code", "?")
        msg = obj["error"].get("message", "?")
        click.echo(f"  ⚠️  {desc}: LFS error [{code}]: {msg}")
        click.echo(f"  💡  Re-upload: ocean cloud upload <repo> '{dest}'")
        return False

    actions = obj.get("actions", {})
    download_actions = actions.get("download", {})
    download_href = download_actions.get("href", "")
    if not download_href:
        click.echo(f"  ⚠️  {desc}: no download URL")
        return False

    resp = requests.get(download_href, stream=True, timeout=7200)
    resp.raise_for_status()
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        with ColoredTqdm(total=file_size, unit="B", unit_scale=True, desc=desc, leave=False) as pbar:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return True


def _download_file_with_lfs(
    repo_id: str,
    file_path: str,
    local_path: str,
    token: str | None,
    dest_dir: Path | None = None,
    revision: str = "master",
    expected_size: int | None = None,
    expected_sha: str | None = None,
):
    """Download a single file by full path, handling LFS transparently.

    Skips download if already cached (file exists + manifest SHA matches).

    Args:
        dest_dir: Destination root dir — used for cache manifest lookups.
            When provided, ``expected_sha`` is also required for cache check.
    """
    local_path_obj = Path(local_path)

    # Skip if already cached via manifest
    if expected_sha and dest_dir and _is_cached(dest_dir, file_path, expected_sha):
        return

    # Skip if file exists with matching size (fast fallback)
    if local_path_obj.exists():
        if expected_size is not None and local_path_obj.stat().st_size == expected_size:
            if expected_sha and dest_dir:
                _mark_cached(dest_dir, file_path, expected_sha)
            return
        local_path_obj.unlink(missing_ok=True)

    git_host = os.getenv("STUDIO_GIT_HOST", _config.GIT_HOST)
    user_name, repo_name = repo_id.split("/")
    url = (
        f"{git_host}/api/v1/repos/{quote(user_name, safe='')}/{quote(repo_name, safe='')}"
        f"/media/{quote(file_path, safe='')}"
    )
    if revision != "master":
        url += f"?ref={quote(revision, safe='')}"

    _download_file(url, str(local_path), token, desc=file_path)

    # Check if downloaded file is an LFS pointer
    if local_path_obj.stat().st_size < 500:
        with open(local_path_obj, "r") as f:
            first_line = f.readline().strip()
        if first_line == "version https://git-lfs.github.com/spec/v1":
            content = local_path_obj.read_text()
            oid_match = re.search(r"oid sha256:([a-f0-9]{64})", content)
            size_match = re.search(r"size (\d+)", content)
            if oid_match and size_match:
                oid = oid_match.group(1)
                file_size = int(size_match.group(1))
                _lfs_download_real(repo_id, oid, file_size, token, str(local_path_obj), desc=file_path)

    # Mark as cached in manifest
    if expected_sha and dest_dir:
        _mark_cached(dest_dir, file_path, expected_sha)


_print_lock = threading.Lock()


def _echo(*args, **kwargs):
    """Thread-safe click.echo."""
    with _print_lock:
        click.echo(*args, **kwargs)


@click.command()
@click.argument("repo_id")
@click.argument("path_in_repo", required=False, default=None)
@click.option("--local-dir", default=".", help="Local directory to save files.")
@click.option("--repo-type", type=click.Choice(["model", "dataset"]), default="dataset")
@click.option("--revision", default="master", help="Branch name.")
@click.option("--token", default=None, help="AI Studio access token.")
@click.option("--include", multiple=True, help="Glob patterns to include.")
@click.option("--exclude", multiple=True, help="Glob patterns to exclude.")
@click.option("--max-workers", default=4, type=int, help="Parallel download workers.")
def download(
    repo_id: str,
    path_in_repo: Optional[str],
    local_dir: str,
    repo_type: str,
    revision: str,
    token: Optional[str],
    include,
    exclude,
    max_workers: int,
):
    """Download files from AI Studio.

    REPO_ID: Format ``username/repo-name``.

    Supports anonymous (public) downloads — no token needed for public repos.
    A token is automatically used when available.

    Examples:

        ocean cloud download PlumBlossom/MyDataset

        ocean cloud download PlumBlossom/MyModel ./model.pdparams --repo-type model
    """
    token = token or get_token_optional()
    _config.validate_repo_id(repo_id)

    dest = Path(local_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if path_in_repo:
        # Download single file
        local_path = dest / Path(path_in_repo).name
        _echo(f"  Downloading {path_in_repo} ...")
        _download_file_with_lfs(repo_id, path_in_repo, str(local_path), token, revision)
        _echo(f"✅ Downloaded to {local_path}")
    else:
        # List all files recursively via git/trees API
        _echo(f"  Listing files in {repo_id} ...")
        files = _list_all_files(repo_id, revision, token)
        if not files:
            _echo("  ⚠️  No files found.")
            return

        _echo(f"  Found {len(files)} file(s), downloading with {max_workers} workers ...")

        # Resolve correct repo-relative paths via Contents API reference
        try:
            root_items = [
                item["name"] for item in list_files(repo_id, repo_type=repo_type, revision=revision, token=token)
            ]
            resolved_paths = _resolve_repo_paths(files, root_items)
        except (requests.RequestException, JSONDecodeError, KeyError, IndexError) as e:
            _echo(f"  ⚠️  Failed to resolve repo paths ({e}), falling back to raw paths.")
            resolved_paths = [f["path"] for f in files]
        for entry, local_rel in zip(files, resolved_paths):
            entry["_local_rel"] = local_rel

        def _download_one(entry: dict) -> tuple[str, bool]:
            """Download a single file. Returns (path, success)."""
            local_rel = entry["_local_rel"]
            local_path = dest / local_rel
            expected_size = entry.get("size")
            expected_sha = entry.get("sha")
            try:
                _download_file_with_lfs(
                    repo_id,
                    local_rel,
                    str(local_path),
                    token,
                    dest_dir=dest,
                    revision=revision,
                    expected_size=expected_size,
                    expected_sha=expected_sha,
                )
                return local_rel, True
            except Exception as e:
                _echo(f"  ✗ {local_rel}")
                _echo(f"    {e}")
                traceback.print_exc()
                return local_rel, False

        success = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_download_one, entry): entry for entry in files}
            for future in as_completed(futures):
                file_path, ok = future.result()
                if ok:
                    success += 1
                else:
                    failed += 1
                    _echo(f"  ✗ {file_path}")

        _echo(f"  ✅ {success} succeeded, ❌ {failed} failed")

    _echo(f"✅ Downloaded from {repo_id}")


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
    token = token or get_token_optional()
    _config.validate_repo_id(repo_id)
    dest_dir = Path(local_dir).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    local_path = dest_dir / Path(path_in_repo).name
    _download_file_with_lfs(repo_id, path_in_repo, str(local_path), token, revision)
    return str(local_path)
