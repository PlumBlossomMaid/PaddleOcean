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
import threading
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

# ── Gitea API serialization ─────────────────────────────────────────
# Gitea returns 500 under concurrent requests; serialize all API calls.
_gitea_lock = threading.Lock()

# ── Retryable network exceptions ────────────────────────────────────
_RETRYABLE_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.SSLError,
)


def _retry_call(fn, max_retries=3, desc=""):
    """Retry a callable with exponential backoff.

    Only retries for network-level errors (timeout / connection / SSL).
    Follows the pattern in ``_sts_multipart_upload._upload_with_retry``.

    Args:
        fn: Zero-argument callable to retry.
        max_retries: Maximum retry attempts (default 3).
        desc: Description for progress messages.

    Returns:
        The return value of ``fn()`` on success.

    Raises:
        The last exception if all retries are exhausted.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except _RETRYABLE_EXCEPTIONS:
            if attempt < max_retries:
                wait = min(2**attempt, 30)
                click.echo(f"    ⚠️  {desc} (attempt {attempt}/{max_retries}), retry in {wait}s...")
                time.sleep(wait)
            else:
                raise


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
    """Call AI Studio Gitea API (serialized via module-level lock).

    Retries on network errors (timeout / connection / SSL) and transient
    500 responses, matching the backoff pattern in ``_sts_multipart_upload``.
    """
    host = os.getenv("STUDIO_GIT_HOST", _config.GIT_HOST)
    url = f"{host}{path}"
    headers = _header_fill(token)
    if content_type:
        headers["Content-Type"] = content_type
        headers["Accept"] = content_type
    body = json.dumps(data) if data is not None else None

    for attempt in range(1, 4):
        try:
            with _gitea_lock:
                resp = requests.request(method, url, headers=headers, data=body, timeout=60)
        except _RETRYABLE_EXCEPTIONS as e:
            if attempt < 3:
                wait = min(2**attempt, 30)
                click.echo(f"    ⚠️  Gitea {method} (attempt {attempt}/3 — {type(e).__name__}), retry in {wait}s...")
                time.sleep(wait)
                continue
            raise click.ClickException(f"Git API timeout (retries exhausted): {e}")

        if resp.status_code in (200, 201):
            return resp.json()

        if resp.status_code == 500 and attempt < 3:
            wait = min(2**attempt, 30)
            click.echo(f"    ⚠️  Gitea {method} 500 (attempt {attempt}/3), retry in {wait}s...")
            time.sleep(wait)
            continue

        raise click.ClickException(f"Git API error [{resp.status_code}]: {resp.text[:200]}")


def _check_file_exists(repo_id: str, path_in_repo: str, revision: str, token: str):
    """Check if a file exists in the repo, return sha if it does."""
    host = os.getenv("STUDIO_GIT_HOST", _config.GIT_HOST)
    url = f"{host}/api/v1/repos/{repo_id}/contents/{path_in_repo}?ref={revision}"

    def _query():
        with _gitea_lock:
            resp = requests.get(url, headers=_header_fill(token), timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and "sha" in data:
                return data["sha"]
        return None

    return _retry_call(_query, desc=f"check exists {path_in_repo}")


# ── BOS upload helpers ──────────────────────────────────────────────


def _http_put(url: str, local_path: str, desc: str) -> None:
    """Upload a file to BOS via streaming HTTP PUT.

    Retries on network errors (timeout / connection / SSL) with
    exponential backoff up to 3 attempts.
    """
    file_size = os.path.getsize(local_path)

    def _do_upload():
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

    _retry_call(_do_upload, desc=f"BOS PUT {desc}")


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


def _lfs_upload_content(
    repo_id: str,
    path_in_repo: str,
    local_path: str,
    revision: str,
    token: str,
) -> Optional[str]:
    """Upload LFS file content to BOS storage (no Gitea pointer commit).

    Returns the SHA256 hex digest if successful, or None if skipped.

    Flow:
        1. Compute SHA256 + size
        2. Call LFS batch API → get STS token or upload_href
        3. Upload to BOS (multipart with STS, or simple PUT to href)
    """
    file_size = os.path.getsize(local_path)
    sha = _sha256(local_path)
    user_name, repo_name = repo_id.split("/")

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
        return None

    obj = resp["objects"][0]
    actions = obj.get("actions", {})

    if actions:
        upload_info = actions["upload"]
        upload_href = upload_info.get("href", "")
        sts_token = upload_info.get("sts_token", {})

        desc = f"{path_in_repo} ({file_size / 1024 / 1024:.1f} MB)"

        if sts_token and sts_token.get("bosHost") and file_size > 5 * 1024**3:
            _sts_multipart_upload(sts_token, local_path, desc)
        else:
            _http_put(upload_href, local_path, desc)
    else:
        click.echo(f"  ⏭️  {path_in_repo}: content already exists on remote (reusing hash).")

    return sha


# ── Regular file flow ────────────────────────────────────────────────


def _regular_upload_content(
    local_path: str,
) -> None:
    """Validate a small (non-LFS) file is ready for batch commit.

    Raises if the file is too large for the Gitea contents API.
    """
    file_size = os.path.getsize(local_path)
    if file_size > 5 * 1024 * 1024:
        raise click.ClickException(f"{local_path}: {file_size} bytes exceeds the 5MB limit for non-LFS upload.")


# ── Main upload dispatch ─────────────────────────────────────────────


def _get_lfs_map(
    repo_id: str,
    revision: str,
    token: str,
    items: list[tuple[str, str]],
) -> dict[str, bool]:
    """Query AI Studio's preupload API to determine which files must be LFS.

    Respects the repo's ``.gitattributes`` LFS rules (e.g. ``*.zip`` →
    ``filter=lfs``).  Files not returned by the API default to size-based
    heuristics (>5MB → LFS).

    Returns a dict mapping ``path_in_repo`` → ``is_lfs``.
    """
    user_name, repo_name = repo_id.split("/")
    host = os.getenv("STUDIO_GIT_HOST", _config.GIT_HOST)

    # Build path list
    path_list = [rel_path for rel_path, _ in items]
    url = (
        f"{host}/api/v1/repos/"
        f"{quote(user_name, safe='')}/{quote(repo_name, safe='')}/preupload/{quote(revision, safe='')}"
    )
    params = {"files": [{"path": p} for p in path_list]}
    headers = _header_fill(token)

    try:
        resp = requests.post(url, headers=headers, json=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if "files" in data and isinstance(data["files"], list):
                lfs_map = {}
                for entry in data["files"]:
                    path = entry.get("path")
                    is_lfs = entry.get("lfs", False)
                    if path:
                        lfs_map[path] = is_lfs
                return lfs_map
    except Exception:
        pass

    # Fallback: size-based heuristic
    return {}


def _upload_item(
    repo_id: str,
    path_in_repo: str,
    local_path: str,
    revision: str,
    token: str,
    is_lfs: bool,
) -> Optional[tuple]:
    """Upload a single file's content to BOS (LFS) or validate for batch commit.

    Returns ``(path_in_repo, local_path, is_lfs, sha256)`` tuple,
    or ``None`` if the file should be skipped.
    """
    file_size = os.path.getsize(local_path)
    lfs_threshold = int(os.environ.get("OCEAN_UPLOAD_LFS_THRESHOLD", 5 * 1024 * 1024))
    if not is_lfs and file_size > lfs_threshold:
        is_lfs = True  # size-based fallback even if API missed it

    if is_lfs:
        sha = _lfs_upload_content(repo_id, path_in_repo, local_path, revision, token)
        if sha is None:
            return None
        return (path_in_repo, local_path, True, sha)
    else:
        _regular_upload_content(local_path)
        sha = _sha256(local_path)
        return (path_in_repo, local_path, False, sha)


# ── Batch commit ─────────────────────────────────────────────────────


def _batch_commit(
    repo_id: str,
    revision: str,
    token: str,
    commit_message: Optional[str],
    file_quads: list[tuple[str, str, bool, str]],
) -> None:
    """Batch commit LFS pointers and small files to Gitea in one request.

    Uses the batch endpoint ``POST /api/v1/repos/{repo_id}/contents``
    with ``files: [...]`` — avoids the Gitea 500 error that occurs
    when creating LFS pointers via the single-file endpoint.

    Args:
        repo_id: ``username/repo-name``.
        revision: Branch name.
        token: AI Studio access token.
        commit_message: Optional commit message.
        file_quads: List of ``(path_in_repo, local_path, is_lfs, sha256)`` tuples.
    """
    host = os.getenv("STUDIO_GIT_HOST", _config.GIT_HOST)
    headers = _header_fill(token)
    url = f"{host}/api/v1/repos/{repo_id}/contents"

    author = {"name": "ocean", "email": "ocean@paddleocean.dev"}
    committer = author

    files = []
    for path_in_repo, local_path, is_lfs, sha256 in file_quads:
        if is_lfs:
            size = os.path.getsize(local_path)
            pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha256}\nsize {size}\n"
            content_b64 = base64.b64encode(pointer.encode()).decode()
        else:
            with open(local_path, "rb") as f:
                content_b64 = base64.b64encode(f.read()).decode()

        existing_sha = _check_file_exists(repo_id, path_in_repo, revision, token)
        entry = {
            "lfsPointer": is_lfs,
            "path": path_in_repo,
            "content": content_b64,
            "operation": "update" if existing_sha else "create",
        }
        if existing_sha:
            entry["sha"] = existing_sha
        files.append(entry)

    payload = {
        "branch": revision,
        "message": commit_message or f"Upload {len(files)} file(s) via ocean",
        "author": author,
        "committer": committer,
        "files": files,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code // 100 == 2:
        click.echo(f"  ✅ Committed {len(files)} file(s).")
    else:
        raise click.ClickException(f"Batch commit failed [{resp.status_code}]: {resp.text[:200]}")


@click.command()
@click.argument("repo_id")
@click.argument("local_paths", nargs=-1, type=click.Path(exists=True))
@click.option("--path-in-repo", default=None, help="Target path in repo.")
@click.option("--repo-type", type=click.Choice(["model", "dataset"]), default="dataset")
@click.option("--revision", default="master", help="Branch name.")
@click.option("--token", default=None, help="AI Studio access token.")
@click.option("--max-workers", default=4, type=int, help="Parallel upload workers.")
@click.option("--commit-message", default=None, help="Commit message.")
def upload(
    repo_id: str,
    local_paths: tuple[str, ...],
    path_in_repo: Optional[str],
    repo_type: str,
    revision: str,
    token: Optional[str],
    max_workers: int,
    commit_message: Optional[str],
):
    """Upload file(s) or folder(s) to AI Studio.

    REPO_ID: Format ``username/repo-name``.

    LOCAL_PATHS: One or more local files or folders to upload.
                   Pass multiple paths separated by spaces.

    Examples:

        ocean cloud upload PlumBlossom/MyDataset ./data.zip

        ocean cloud upload PlumBlossom/MyDataset ./part1.zip ./part2.zip ./part3.zip

        ocean cloud upload PlumBlossom/MyModel ./checkpoints/ --repo-type model
    """
    upload_folder(
        repo_id=repo_id,
        local_paths=local_paths,
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
        local_paths=[local_path],
        path_in_repo=path_in_repo,
        repo_type=repo_type,
        revision=revision,
        token=token,
        max_workers=1,
        commit_message=commit_message,
    )


def upload_folder(
    repo_id: str,
    local_paths: str | list[str] | tuple[str, ...],
    path_in_repo: Optional[str] = None,
    repo_type: str = "dataset",
    revision: str = "master",
    token: Optional[str] = None,
    max_workers: int = 4,
    commit_message: Optional[str] = None,
) -> None:
    """Upload file(s) or folder(s) to AI Studio.

    Args:
        repo_id: ``username/repo-name``.
        local_paths: One or more local files or directories to upload.
        path_in_repo: Target path in the repo.
        repo_type: ``"dataset"`` or ``"model"``.
        revision: Branch name.
        token: AI Studio access token. Falls back to env/login.
        max_workers: Parallel upload threads.
        commit_message: Optional commit message.

    Examples:
        >>> upload_folder("PlumBlossom/MyData", "./data_dir/")
        >>> upload_folder("PlumBlossom/MyData", ["./part1.zip", "./part2.zip"])
    """
    if isinstance(local_paths, str):
        local_paths = [local_paths]

    token = token or get_token()
    _config.validate_repo_id(repo_id)

    items: list[tuple[str, str]] = []
    for lp in local_paths:
        p = Path(lp)
        if p.is_file():
            items.append((path_in_repo or p.name, str(p)))
        else:
            prefix = (path_in_repo or "").strip("/")
            prefix = f"{prefix}/" if prefix else ""
            for f in sorted(p.rglob("*")):
                if f.is_file() and not f.name.startswith("."):
                    rel = f.relative_to(p).as_posix()
                    items.append((prefix + rel, str(f)))

    click.echo(f"  Uploading {len(items)} file(s) to {repo_id}...")
    file_quads: list[tuple[str, str, bool, str]] = []
    errors: list[tuple[str, str]] = []

    # Query repo LFS rules (respects .gitattributes like ``*.zip filter=lfs``)
    lfs_map = _get_lfs_map(repo_id, revision, token, items)

    # Warn about blocked archive extensions on model repos
    _blocked_extensions = (".zip", ".rar", ".7z", ".tar")
    if repo_type == "model":
        blocked = [p for p, _ in items if os.path.splitext(p)[1].lower() in _blocked_extensions]
        if blocked:
            click.echo(
                "  ⚠️  AI Studio model repos block uploading archive files (.zip/.rar/.7z/.tar)\n"
                "        through the Gitea API.  Use --path-in-repo with a different\n"
                "        extension (e.g. filename.bin) to work around this limitation.",
                err=True,
            )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_to_item = {}
        for rel_path, abs_path in items:
            is_lfs = lfs_map.get(rel_path, False)
            fut = pool.submit(_upload_item, repo_id, rel_path, abs_path, revision, token, is_lfs)
            fut_to_item[fut] = (rel_path, abs_path)
        for future in as_completed(fut_to_item):
            rel_path, abs_path = fut_to_item[future]
            try:
                result = future.result()
                if result is not None:
                    file_quads.append(result)
            except Exception as e:
                click.echo(f"  ❌ {rel_path}: {e}")
                errors.append((rel_path, str(e)))

    if errors:
        click.echo(f"\n  Done with {len(errors)} error(s):")
        for rel_path, err in errors:
            click.echo(f"    ❌ {rel_path}: {err}")
        raise click.ClickException(f"{len(errors)} file(s) failed to upload.")

    # Batch commit all uploaded files
    if file_quads:
        click.echo(f"  Committing {len(file_quads)} file(s)...")
        try:
            _batch_commit(repo_id, revision, token, commit_message, file_quads)
        except Exception as e:
            raise click.ClickException(f"Commit failed: {e}")
    else:
        click.echo("  No files to commit.")

    click.echo(f"  ✅ Done. All files uploaded to {repo_id}")
