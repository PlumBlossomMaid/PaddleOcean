"""cloud upload — upload files to AI Studio.

Public API:
    >>> from ocean.cloud import upload_file, upload_folder
    >>> upload_file("user/repo", "/path/to/file.zip", repo_type="dataset")

CLI:
    ocean cloud upload user/repo /path/to/file.zip

Design: ocean maintains its own BOS upload implementation decoupled from
both ``aistudio-sdk`` and ``baidubce``. Large files are uploaded via
direct HTTP/REST calls to BOS — no BCE SDK dependency.
"""

import base64
import hashlib
import hmac
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlparse

import click
import requests

from ocean.cli.cloud import _config
from ocean.cli.cloud.auth import get_token
from ocean.utils.colored_tqdm import ColoredTqdm

__all__ = ["upload"]

# ── BOS signing helpers (BCE auth v1, no SDK dependency) ─────────────


def _get_canonical_time(timestamp: int) -> str:
    """Format timestamp → ``2025-01-15T12:00:00Z``."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def _normalize_string(s: str) -> str:
    """Encode string with ``%XX`` for bytes > 127 and special chars."""
    result = []
    for b in s.encode("utf-8"):
        if b in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.~":
            result.append(chr(b))
        else:
            result.append(f"%{b:02X}")
    return "".join(result)


def _bce_sign(
    access_key: str,
    secret_key: str,
    http_method: str,
    path: str,
    headers: dict,
    params: dict,
    timestamp: Optional[int] = None,
    expiration: int = 1800,
) -> str:
    """BCE Signature V1 — copied algorithm from ``baidubce.auth.bce_v1_signer``."""
    if timestamp is None:
        timestamp = int(time.time())

    headers = {k.lower(): v for k, v in headers.items()}

    # sign_key = HMAC-SHA256(secret, "bce-auth-v1/{ak}/{time}/{exp}")
    sign_key_info = f"bce-auth-v1/{access_key}/{_get_canonical_time(timestamp)}/{expiration}"
    sign_key = hmac.new(
        secret_key.encode(),
        sign_key_info.encode(),
        hashlib.sha256,
    ).hexdigest()

    # canonical URI
    canonical_uri = path

    # canonical query string
    sorted_params = sorted(params.items(), key=lambda x: x[0])
    q_parts = []
    for k, v in sorted_params:
        k_enc = quote(k, safe="")
        v_enc = quote(str(v), safe="") if v else ""
        q_parts.append(f"{k_enc}={v_enc}")
    canonical_querystring = "&".join(q_parts)

    # canonical headers
    signed_headers_set = {"host", "content-md5", "content-length", "content-type"}
    bce_prefix = "x-bce-"
    header_lines = []
    for k, v in sorted(headers.items()):
        if k.startswith(bce_prefix) or k in signed_headers_set:
            header_lines.append(f"{_normalize_string(k)}:{_normalize_string(v)}")
    canonical_headers = "\n".join(header_lines)

    # string to sign
    string_to_sign = f"{http_method}\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}"

    # final signature
    signature = hmac.new(
        sign_key.encode(),
        string_to_sign.encode(),
        hashlib.sha256,
    ).hexdigest()

    # auth header value
    auth = f"bce-auth-v1/{access_key}/{_get_canonical_time(timestamp)}/{expiration}//{signature}"
    return auth


# ── BOS REST client (no baidubce SDK) ────────────────────────────────


class BosRestClient:
    """Upload files to Baidu Object Storage via raw REST API.

    Supports both regular PUT and multipart upload. Uses BCE auth V1
    for request signing — no dependency on ``baidubce`` SDK.
    """

    def __init__(self, bos_host: str, access_key: str, secret_key: str, session_token: str):
        if not bos_host.startswith("http"):
            bos_host = "https://" + bos_host
        self.bos_host = bos_host.rstrip("/")
        self.access_key = access_key
        self.secret_key = secret_key
        self.session_token = session_token

    def _headers(self, extra: Optional[dict] = None) -> dict:
        h = {
            "Host": urlparse(self.bos_host).netloc,
            "x-bce-date": _get_canonical_time(int(time.time())),
            "x-bce-security-token": self.session_token,
        }
        if extra:
            h.update(extra)
        return h

    def _sign_and_send(
        self,
        method: str,
        bucket: str,
        key: str,
        body=None,
        params: Optional[dict] = None,
    ) -> requests.Response:
        # key may contain slashes (e.g. "lfs/c3/63/..."), preserve them in path
        path = f"/{quote(bucket, safe='')}/{quote(key, safe='/')}"
        headers = self._headers()
        if body is not None:
            headers["Content-Type"] = "application/octet-stream"

        params = params or {}
        auth = _bce_sign(
            self.access_key,
            self.secret_key,
            method,
            path,
            headers | {"Host": headers["Host"]},
            params,
        )
        headers["Authorization"] = auth
        url = f"{self.bos_host}{path}"
        if params:
            qs = "&".join(f"{quote(k, safe='')}={quote(str(v), safe='')}" for k, v in sorted(params.items()))
            url = f"{url}?{qs}"
        resp = requests.request(method, url, headers=headers, data=body, timeout=7200)
        return resp

    def initiate_multipart(self, bucket: str, key: str) -> str:
        """POST /{bucket}/{key}?uploads → return upload_id."""
        resp = self._sign_and_send("POST", bucket, key, params={"uploads": ""})
        resp.raise_for_status()
        # Response is XML: <InitiateMultipartUploadResult><uploadId>...</uploadId></InitiateMultipartUploadResult>
        import xml.etree.ElementTree as ET

        return ET.fromstring(resp.text).findtext("uploadId")

    def upload_part(self, bucket: str, key: str, upload_id: str, part_number: int, data: bytes) -> str:
        """PUT /{bucket}/{key}?partNumber=N&uploadId=ID → return ETag."""
        params = {"partNumber": part_number, "uploadId": upload_id}
        headers = self._headers({"Content-Length": str(len(data))})
        path = f"/{quote(bucket, safe='')}/{quote(key, safe='/')}"
        auth = _bce_sign(
            self.access_key,
            self.secret_key,
            "PUT",
            path,
            headers | {"Host": headers["Host"]},
            params,
        )
        headers["Authorization"] = auth
        qs = "&".join(f"{quote(k, safe='')}={quote(str(v), safe='')}" for k, v in sorted(params.items()))
        url = f"{self.bos_host}{path}?{qs}"

        resp = requests.put(url, headers=headers, data=data, timeout=7200)
        resp.raise_for_status()
        return resp.headers.get("ETag", "").strip('"')

    def complete_multipart(self, bucket: str, key: str, upload_id: str, parts: list) -> None:
        """POST /{bucket}/{key}?uploadId=ID with parts list → complete."""
        body = json.dumps({"parts": [{"partNumber": p["partNumber"], "eTag": p["eTag"]} for p in parts]})
        resp = self._sign_and_send("POST", bucket, key, body=body, params={"uploadId": upload_id})
        resp.raise_for_status()

    def abort_multipart(self, bucket: str, key: str, upload_id: str) -> None:
        """DELETE /{bucket}/{key}?uploadId=ID → abort."""
        resp = self._sign_and_send("DELETE", bucket, key, params={"uploadId": upload_id})
        resp.raise_for_status()

    def put_object(self, bucket: str, key: str, local_path: str) -> None:
        """Regular PUT of a file to BOS."""
        file_size = os.path.getsize(local_path)
        path = f"/{quote(bucket, safe='')}/{quote(key, safe='/')}"
        initial_headers = self._headers({"Content-Length": str(file_size), "Content-Type": "application/octet-stream"})
        auth = _bce_sign(
            self.access_key,
            self.secret_key,
            "PUT",
            path,
            initial_headers | {"Host": initial_headers["Host"]},
            {},
        )
        initial_headers["Authorization"] = auth
        url = f"{self.bos_host}{path}"

        def _iter_upload():
            with open(local_path, "rb") as f:
                with ColoredTqdm(
                    total=file_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"  ☁️  {os.path.basename(local_path)}",
                    leave=False,
                ) as pbar:
                    while True:
                        chunk = f.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        yield chunk
                        pbar.update(len(chunk))

        resp = requests.put(url, headers=initial_headers, data=_iter_upload(), timeout=7200)
        resp.raise_for_status()

    def put_super_object(
        self, bucket: str, key: str, local_path: str, chunk_size_mb: int = 5, thread_num: Optional[int] = None
    ) -> None:
        """Multipart upload (split + parallel parts + complete).

        Algorithm matches ``baidubce.services.bos.bos_client.BosClient.put_super_obejct_from_file``,
        but implemented via direct REST API calls — no SDK dependency.
        """
        file_size = os.path.getsize(local_path)
        part_size = chunk_size_mb * 1024 * 1024
        total_parts = (file_size + part_size - 1) // part_size

        # Initiate
        upload_id = self.initiate_multipart(bucket, key)
        click.echo(f"    Multipart: initiated upload_id={upload_id}, {total_parts} parts")

        # Upload parts in parallel
        if thread_num is None:
            thread_num = os.cpu_count() or 4

        part_list = []

        def _upload_single_part(part_number: int, offset: int, size: int):
            with open(local_path, "rb") as f:
                f.seek(offset)
                data = f.read(size)
            etag = self.upload_part(bucket, key, upload_id, part_number, data)
            return {"partNumber": part_number, "eTag": etag, "size": size}

        with ColoredTqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=f"  ☁️  {os.path.basename(local_path)}",
            leave=False,
        ) as pbar:
            with ThreadPoolExecutor(max_workers=thread_num) as pool:
                futures = []
                offset = 0
                for part_number in range(1, total_parts + 1):
                    size = min(part_size, file_size - offset)
                    futures.append(pool.submit(_upload_single_part, part_number, offset, size))
                    offset += size

                for future in as_completed(futures):
                    result = future.result()
                    part_list.append(result)
                    pbar.update(result["size"])

        # Sort by part number
        part_list.sort(key=lambda x: x["partNumber"])

        if len(part_list) != total_parts:
            self.abort_multipart(bucket, key, upload_id)
            raise click.ClickException(f"Multipart upload aborted: expected {total_parts} parts, got {len(part_list)}")

        # Complete
        self.complete_multipart(bucket, key, upload_id, part_list)
        click.echo(f"    Multipart: completed ({total_parts} parts)")


# ── helpers ──────────────────────────────────────────────────────────


def _sha256(filepath: str, desc: str = "") -> str:
    h = hashlib.sha256()
    file_size = os.path.getsize(filepath)
    with open(filepath, "rb") as f:
        with ColoredTqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=f"  🔑 {desc}" if desc else "  🔑 SHA256",
            leave=False,
        ) as pbar:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
                pbar.update(len(chunk))
    return h.hexdigest()


def _header_fill(token: str, extra: Optional[dict] = None) -> dict:
    h = {
        "Content-Type": "application/json",
        "Authorization": f"token {token}",
    }
    if extra:
        h.update(extra)
    return h


def _git_api(method: str, path: str, token: str, data=None, content_type: Optional[str] = None) -> dict:
    """Call AI Studio Gitea API."""
    host = os.getenv("STUDIO_GIT_HOST", _config.GIT_HOST)
    url = f"{host}{path}"
    headers = _header_fill(token)
    if content_type:
        headers["Content-Type"] = content_type
        headers["Accept"] = content_type
    body = json.dumps(data) if data is not None else None
    resp = requests.request(method, url, headers=headers, data=body, timeout=30)
    if resp.status_code in (200, 201):
        return resp.json()
    raise click.ClickException(f"Git API error [{resp.status_code}]: {resp.text[:200]}")


def _check_file_exists(repo_id: str, path_in_repo: str, revision: str, token: str):
    """Check if a file exists in the repo, return sha if it does."""
    host = os.getenv("STUDIO_GIT_HOST", _config.GIT_HOST)
    url = f"{host}/api/v1/repos/{repo_id}/contents/{path_in_repo}?ref={revision}"
    resp = requests.get(url, headers=_header_fill(token), timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, dict) and "sha" in data:
            return data["sha"]
    return None


# ── BOS upload helpers ──────────────────────────────────────────────


def _http_put(url: str, local_path: str, desc: str) -> None:
    """Upload a file to BOS via streaming HTTP PUT."""
    file_size = os.path.getsize(local_path)

    def _iter_upload():
        with open(local_path, "rb") as f:
            with ColoredTqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                desc=f"  ☁️  {desc}",
                leave=False,
            ) as pbar:
                while True:
                    chunk = f.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    yield chunk
                    pbar.update(len(chunk))

    resp = requests.put(
        url,
        data=_iter_upload(),
        headers={"Content-Type": "application/octet-stream"},
        timeout=7200,
    )
    resp.raise_for_status()


def _sts_multipart_upload(sts_token: dict, local_path: str, desc: str) -> None:
    """Upload via STS multipart with progress tracking.

    Uses ``aistudio_sdk.utils.bos_sdk.sts_client`` for the BOS client,
    but implements the multipart loop with a ColoredTqdm progress bar.
    """
    try:
        from aistudio_sdk.utils.bos_sdk import sts_client
    except ImportError:
        raise click.ClickException("STS multipart upload requires aistudio-sdk: pip install aistudio-sdk")

    bos_host = sts_token.get("bosHost", "")
    if not bos_host.startswith("http"):
        bos_host = "https://" + bos_host
    client = sts_client(
        bos_host,
        sts_token.get("accessKeyId"),
        sts_token.get("secretAccessKey"),
        sts_token.get("sessionToken"),
    )
    bucket = sts_token.get("bucketName", "")
    key = sts_token.get("key", "")
    file_size = os.path.getsize(local_path)
    chunk_size_mb = int(os.environ.get("OCEAN_UPLOAD_CHUNK_SIZE_MB", 5))
    thread_num = int(os.environ.get("OCEAN_UPLOAD_THREAD_NUM", os.cpu_count() or 4))
    part_size = chunk_size_mb * 1024 * 1024
    total_parts = (file_size + part_size - 1) // part_size

    click.echo(f"  (STS multipart: {total_parts} parts, {chunk_size_mb}MB each, {thread_num} threads)")

    # 1. Initiate
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _upload_with_retry(pn, sz, offset, max_retries=5):
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.upload_part_from_file(
                    bucket,
                    key,
                    upload_id,
                    pn,
                    sz,
                    local_path,
                    offset,
                )
                return resp.metadata.etag
            except Exception:
                if attempt < max_retries:
                    wait = min(2**attempt, 30)
                    click.echo(f"    ⚠️  part {pn} failed (attempt {attempt}), retry in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    upload_id = client.initiate_multipart_upload(bucket, key).upload_id

    # 2. Upload parts in parallel with progress
    part_list = []
    with ColoredTqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        desc=f"  ☁️  [STS] {desc}",
        leave=False,
    ) as pbar:
        with ThreadPoolExecutor(max_workers=thread_num) as pool:
            fut_to_part = {}
            offset = 0
            for pn in range(1, total_parts + 1):
                sz = min(part_size, file_size - offset)
                fut = pool.submit(_upload_with_retry, pn, sz, offset)
                fut_to_part[fut] = (pn, sz)
                offset += sz

            for fut in as_completed(fut_to_part):
                etag = fut.result()
                pn, sz = fut_to_part[fut]
                part_list.append({"partNumber": pn, "eTag": etag})
                pbar.update(sz)

    # 3. Complete
    part_list.sort(key=lambda x: x["partNumber"])
    client.complete_multipart_upload(bucket, key, upload_id, part_list)


# ── LFS flow ────────────────────────────────────────────────────────


def _lfs_upload_file(
    repo_id: str,
    path_in_repo: str,
    local_path: str,
    revision: str,
    token: str,
    commit_message: Optional[str] = None,
):
    """Upload a single LFS file to AI Studio.

    Flow:
        1. Compute SHA256 + size
        2. Call LFS batch API → get STS token or upload_href
        3. Upload to BOS (multipart with STS, or simple PUT to href)
        4. Upload LFS pointer to Gitea
    """
    file_size = os.path.getsize(local_path)
    sha = _sha256(local_path)
    user_name, repo_name = repo_id.split("/")

    # Step 1: Get upload access (LFS batch API needs ``vnd.git-lfs+json`` Content-Type)
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
        content_type="application/vnd.git-lfs+json",
    )

    if not resp.get("objects"):
        click.echo(f"  ⚠️  {path_in_repo}: no upload actions returned, skipping.")
        return

    obj = resp["objects"][0]
    actions = obj.get("actions", {})

    if actions:
        upload_info = actions["upload"]
        upload_href = upload_info.get("href", "")
        sts_token = upload_info.get("sts_token", {})

        # Step 2: Upload file content to BOS
        desc = f"{path_in_repo} ({file_size / 1024 / 1024:.1f} MB)"

        if sts_token and sts_token.get("bosHost") and file_size > 5 * 1024**3:
            # Files >5GB: BOS single PUT limit is 5GB, use STS multipart directly.
            _sts_multipart_upload(sts_token, local_path, desc)
        else:
            _http_put(upload_href, local_path, desc)
    else:
        click.echo(f"  ⏭️  {path_in_repo}: content already exists on remote (reusing hash).")

    # Step 3: Upload LFS pointer to Gitea (needed even when content already exists)
    pointer_content = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {file_size}\n"
    pointer_b64 = base64.b64encode(pointer_content.encode()).decode()

    existing_sha = _check_file_exists(repo_id, path_in_repo, revision, token)
    method = "PUT" if existing_sha else "POST"
    payload = {
        "branch": revision,
        "content": pointer_b64,
        "lfsPointer": True,
    }
    if existing_sha:
        payload["sha"] = existing_sha
    if commit_message:
        payload["message"] = commit_message

    _git_api(method, f"/api/v1/repos/{repo_id}/contents/{path_in_repo}", token, data=payload)
    click.echo(f"  ✅ {path_in_repo} uploaded.")


# ── Regular file flow ────────────────────────────────────────────────


def _regular_upload_file(
    repo_id: str,
    path_in_repo: str,
    local_path: str,
    revision: str,
    token: str,
    commit_message: Optional[str] = None,
):
    """Upload a small file (non-LFS) via Gitea API."""
    with open(local_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode()

    existing_sha = _check_file_exists(repo_id, path_in_repo, revision, token)
    method = "PUT" if existing_sha else "POST"
    payload = {
        "branch": revision,
        "content": content_b64,
        "lfs": False,
    }
    if existing_sha:
        payload["sha"] = existing_sha
    if commit_message:
        payload["message"] = commit_message

    _git_api(method, f"/api/v1/repos/{repo_id}/contents/{path_in_repo}", token, data=payload)


# ── Main upload dispatch ─────────────────────────────────────────────


def _upload_item(
    repo_id: str,
    path_in_repo: str,
    local_path: str,
    revision: str,
    token: str,
    commit_message: Optional[str] = None,
):
    """Upload a single file, choosing LFS or regular flow."""
    file_size = os.path.getsize(local_path)

    # Files larger than 5MB go through LFS. Small files use Gitea API directly.
    is_lfs = file_size > int(os.environ.get("OCEAN_UPLOAD_LFS_THRESHOLD", 5 * 1024 * 1024))

    if is_lfs:
        _lfs_upload_file(repo_id, path_in_repo, local_path, revision, token, commit_message)
    else:
        _regular_upload_file(repo_id, path_in_repo, local_path, revision, token, commit_message)


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

    click.echo(f"  Uploading {len(items)} file(s) to {repo_id}...")
    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_to_item = {
            pool.submit(_upload_item, repo_id, rel_path, abs_path, revision, token, commit_message): (
                rel_path,
                abs_path,
            )
            for rel_path, abs_path in items
        }
        for future in as_completed(fut_to_item):
            rel_path, abs_path = fut_to_item[future]
            try:
                future.result()
            except Exception as e:
                click.echo(f"  ❌ {rel_path}: {e}")
                errors.append((rel_path, str(e)))

    if errors:
        click.echo(f"\n  Done with {len(errors)} error(s):")
        for rel_path, err in errors:
            click.echo(f"    ❌ {rel_path}: {err}")
        raise click.ClickException(f"{len(errors)} file(s) failed to upload.")
    else:
        click.echo(f"  ✅ Done. All files uploaded to {repo_id}")
