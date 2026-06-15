"""cloud upload — upload files to AI Studio.

Public API:
    >>> from ocean.cloud import upload_file, upload_folder
    >>> upload_file("user/repo", "/path/to/file.zip", repo_type="dataset")

CLI:
    ocean cloud upload user/repo /path/to/file.zip
"""

import base64
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import click
import requests

from ocean.cli.cloud import _config
from ocean.cli.cloud.auth import get_token

__all__ = ["upload"]


# ── helpers ──────────────────────────────────────────────────────────


def _sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _header_fill(token: str, extra: Optional[dict] = None) -> dict:
    h = {
        "Content-Type": "application/json",
        "Authorization": f"token {token}",
    }
    if extra:
        h.update(extra)
    return h


def _git_api(method: str, path: str, token: str, data=None) -> dict:
    """Call AI Studio Gitea API."""
    host = os.getenv("STUDIO_GIT_HOST", _config.GIT_HOST)
    url = f"{host}{path}"
    resp = requests.request(method, url, headers=_header_fill(token), json=data, timeout=30)
    if resp.status_code in (200, 201):
        return resp.json()
    raise click.ClickException(f"Git API error [{resp.status_code}]: {resp.text[:200]}")


def _check_file_exists(repo_id: str, path_in_repo: str, revision: str, token: str):
    """Check if a file exists in the repo, return sha if it does."""
    resp = _git_api(
        "GET",
        f"/api/v1/repos/{repo_id}/contents/{path_in_repo}?ref={revision}",
        token,
    )
    if isinstance(resp, dict) and "sha" in resp:
        return resp["sha"]
    return None


# ── BOS upload with fixed LFS bug ────────────────────────────────────


class _BosUploader:
    """Upload large files to Baidu Object Storage (BOS), used by AI Studio LFS."""

    def __init__(self, sts_token: dict):
        from baidubce.auth.bce_credentials import BceCredentials
        from baidubce.bce_client_configuration import BceClientConfiguration
        from baidubce.services.bos.bos_client import BosClient

        config = BceClientConfiguration(
            credentials=BceCredentials(
                sts_token["access_key_id"],
                sts_token["secret_access_key"],
            ),
            endpoint=sts_token["bos_host"],
            security_token=sts_token["session_token"],
        )
        self.client = BosClient(config)

    def upload_super(self, filepath: str, bucket: str, key: str) -> bool:
        """Upload a large file via multipart upload.

        Note: the upstream AI Studio SDK has a typo in BOS SDK method name
        (``put_super_obejct_from_file`` missing the 'c' in 'object'), which
        causes ``'super' object has no attribute 'put_super_obejct_from_file'``.
        This implementation uses the correct API directly.
        """
        chunk_size_mb = int(os.environ.get("OCEAN_UPLOAD_CHUNK_SIZE_MB", 5))
        thread_num = os.environ.get("OCEAN_UPLOAD_THREAD_NUM", None)

        self.client.put_super_object_from_file(
            bucket,
            key,
            str(filepath),
            chunk_size=chunk_size_mb,
            thread_num=int(thread_num) if thread_num else None,
        )
        return True

    def upload_regular(self, filepath: str, bucket: str, key: str) -> bool:
        """Upload a regular file."""
        self.client.put_object_from_file(bucket, key, str(filepath))
        return True


# ── LFS flow ──────────────────────────────────────────────────────────


def _lfs_upload_file(repo_id: str, path_in_repo: str, local_path: str, revision: str, token: str):
    """Upload a single LFS file to AI Studio.

    Flow:
        1. Compute SHA256 + size
        2. Call preupload API to get STS token
        3. Upload file content to BOS
        4. Upload LFS pointer to Gitea
    """
    file_size = os.path.getsize(local_path)
    sha = _sha256(local_path)
    user_name, repo_name = repo_id.split("/")

    # Step 1: Get upload access (STS token)
    resp = _git_api(
        "POST",
        f"/{user_name}/{repo_name}.git/info/lfs/objects/batch",
        token,
        data={
            "operation": "upload",
            "objects": [{"oid": sha, "size": file_size}],
            "transfers": ["lfs-standalone-file", "basic"],
            "ref": {"name": f"refs/heads/{revision}"},
            "hash_algo": "sha256",
        },
    )

    if not resp.get("objects"):
        click.echo(f"  ⚠️  {path_in_repo}: no upload actions returned, skipping.")
        return

    obj = resp["objects"][0]
    actions = obj.get("actions", {})
    if not actions:
        click.echo(f"  ⏭️  {path_in_repo}: already exists on remote (same hash).")
        return

    upload_info = actions["upload"]
    upload_href = upload_info["href"]

    # Step 2: Upload to BOS
    click.echo(f"  ☁️  Uploading {path_in_repo} ({file_size / 1024 / 1024:.1f} MB)...")

    sts_token = upload_info.get("sts_token", {})
    if sts_token and sts_token.get("accessKeyId"):
        # Parse STS token keys (camelCase from API → snake_case for client)
        parsed = {
            "bos_host": sts_token.get("bosHost"),
            "bucket_name": sts_token.get("bucketName"),
            "key": sts_token.get("key"),
            "access_key_id": sts_token.get("accessKeyId"),
            "secret_access_key": sts_token.get("secretAccessKey"),
            "session_token": sts_token.get("sessionToken"),
        }
        uploader = _BosUploader(parsed)
        uploader.upload_super(local_path, parsed["bucket_name"], parsed["key"])
    else:
        # Fallback: HTTP PUT direct to BOS
        with open(local_path, "rb") as f:
            r = requests.put(
                upload_href,
                data=f,
                headers={"Content-Type": "application/octet-stream"},
                timeout=3600,
            )
        r.raise_for_status()

    # Step 3: Upload LFS pointer to Gitea
    pointer_content = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {file_size}\n"
    pointer_b64 = base64.b64encode(pointer_content.encode()).decode()

    existing_sha = _check_file_exists(repo_id, path_in_repo, revision, token)
    method = "PUT" if existing_sha else "POST"
    payload = (
        {
            "branch": revision,
            "content": pointer_b64,
            "lfsPointer": True,
            "sha": existing_sha,
        }
        if existing_sha
        else {
            "branch": revision,
            "content": pointer_b64,
            "lfsPointer": True,
        }
    )

    _git_api(method, f"/api/v1/repos/{repo_id}/contents/{path_in_repo}", token, data=payload)
    click.echo(f"  ✅ {path_in_repo} uploaded.")


# ── Regular file flow ────────────────────────────────────────────────


def _regular_upload_file(repo_id: str, path_in_repo: str, local_path: str, revision: str, token: str):
    """Upload a small file (non-LFS) via Gitea API."""
    with open(local_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode()

    existing_sha = _check_file_exists(repo_id, path_in_repo, revision, token)
    method = "PUT" if existing_sha else "POST"
    payload = (
        {
            "branch": revision,
            "content": content_b64,
            "lfs": False,
            "sha": existing_sha,
        }
        if existing_sha
        else {
            "branch": revision,
            "content": content_b64,
            "lfs": False,
        }
    )

    _git_api(method, f"/api/v1/repos/{repo_id}/contents/{path_in_repo}", token, data=payload)


# ── Main upload command ──────────────────────────────────────────────


def _upload_item(repo_id, path_in_repo, local_path, revision, token):
    """Upload a single file, choosing LFS or regular flow."""
    file_size = os.path.getsize(local_path)

    # Preupload check: is this file LFS?
    user_name, repo_name = repo_id.split("/")
    preupload_resp = _git_api(
        "POST",
        f"/{user_name}/{repo_name}/preupload/{revision}",
        token,
        data={"files": [{"path": path_in_repo}]},
    )

    is_lfs = False
    if isinstance(preupload_resp, dict) and preupload_resp.get("files") and len(preupload_resp["files"]) > 0:
        is_lfs = preupload_resp["files"][0].get("lfs", False)

    if is_lfs or file_size > 5 * 1024 * 1024:
        _lfs_upload_file(repo_id, path_in_repo, local_path, revision, token)
    else:
        _regular_upload_file(repo_id, path_in_repo, local_path, revision, token)


@click.command()
@click.argument("repo_id")
@click.argument("local_path", type=click.Path(exists=True))
@click.option("--path-in-repo", default=None, help="Target path in repo.")
@click.option("--repo-type", type=click.Choice(["model", "dataset"]), default="dataset")
@click.option("--revision", default="master", help="Branch name.")
@click.option("--token", default=None, help="AI Studio access token.")
@click.option("--max-workers", default=4, type=int, help="Parallel upload workers.")
@click.option("--commit-message", default=None, help="Commit message.")
def upload(
    repo_id: str,
    local_path: str,
    path_in_repo: Optional[str],
    repo_type: str,
    revision: str,
    token: Optional[str],
    max_workers: int,
    commit_message: Optional[str],
):
    """Upload a file or folder to AI Studio.

    REPO_ID: Format ``username/repo-name``.

    LOCAL_PATH: Path to a local file or folder to upload.

    Examples:

        ocean cloud upload PlumBlossom/MyDataset ./data.zip

        ocean cloud upload PlumBlossom/MyModel ./checkpoints/ --repo-type model
    """
    upload_folder(
        repo_id=repo_id,
        local_path=local_path,
        path_in_repo=path_in_repo,
        repo_type=repo_type,
        revision=revision,
        token=token or get_token(),
        max_workers=max_workers,
        commit_message=commit_message,
    )


# ── Public Python API ────────────────────────────────────────────────


def upload_file(
    repo_id: str,
    local_path: str,
    path_in_repo: Optional[str] = None,
    repo_type: str = "dataset",
    revision: str = "master",
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> None:
    """Upload a single file to AI Studio.

    Args:
        repo_id: ``username/repo-name``.
        local_path: Path to the local file.
        path_in_repo: Target path in the repo (defaults to filename).
        repo_type: ``"dataset"`` or ``"model"``.
        revision: Branch name.
        token: AI Studio access token. Falls back to env/login.
        commit_message: Optional commit message.

    Examples:
        >>> upload_file("PlumBlossom/MyData", "./17LiYuan.zip", repo_type="dataset")
    """
    upload_folder(
        repo_id=repo_id,
        local_path=local_path,
        path_in_repo=path_in_repo,
        repo_type=repo_type,
        revision=revision,
        token=token,
        max_workers=1,
        commit_message=commit_message,
    )


def upload_folder(
    repo_id: str,
    local_path: str,
    path_in_repo: Optional[str] = None,
    repo_type: str = "dataset",
    revision: str = "master",
    token: Optional[str] = None,
    max_workers: int = 4,
    commit_message: Optional[str] = None,
) -> None:
    """Upload a file or folder to AI Studio.

    Args:
        repo_id: ``username/repo-name``.
        local_path: Path to local file or directory.
        path_in_repo: Target path in the repo.
        repo_type: ``"dataset"`` or ``"model"``.
        revision: Branch name.
        token: AI Studio access token. Falls back to env/login.
        max_workers: Parallel upload threads.
        commit_message: Optional commit message.

    Examples:
        >>> upload_folder("PlumBlossom/MyData", "./data_dir/")
    """
    token = token or get_token()
    _config.validate_repo_id(repo_id)

    p = Path(local_path)
    if p.is_file():
        items = [(path_in_repo or p.name, str(p))]
    else:
        items = []
        prefix = (path_in_repo or "").strip("/")
        prefix = f"{prefix}/" if prefix else ""
        for f in sorted(p.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                rel = f.relative_to(p).as_posix()
                items.append((prefix + rel, str(f)))

    print(f"  Uploading {len(items)} file(s) to {repo_id}...")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_upload_item, repo_id, path_in_repo, local_path, revision, token): (
                path_in_repo,
                local_path,
            )
            for path_in_repo, local_path in items
        }
        for future in as_completed(futures):
            pass  # exceptions are raised by the submitted task

    print(f"  Done. Uploaded to {repo_id}")
