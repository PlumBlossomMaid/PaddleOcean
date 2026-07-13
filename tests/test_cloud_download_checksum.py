"""Tests for cloud download checksum verification.

Regression: the git/trees API reports a file's ``sha`` as a *git blob SHA-1*
(``sha1("blob <size>\\0" + content)``), but the downloader verified regular
files by computing a *sha256 of the content* and comparing it to that blob sha.
The two never match, so every regular file was flagged as corrupt and deleted.

For LFS files it was worse: the ``media`` endpoint returns the *dereferenced
content* (not the small pointer), whose only correct checksum is the LFS
``oid`` (a sha256) — the git/trees sha/size describe the ~134-byte pointer and
cannot verify the content.

These tests lock in:
  * ``_git_blob_sha1`` matches what ``git hash-object`` produces,
  * regular-file verification uses git-blob-SHA1 (pass on match, delete on
    mismatch),
  * LFS verification uses the oid (sha256 of the real content),
  * the size-based fast-skip uses the real LFS size, not the pointer size.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from ocean.cli.cloud import download as dl


# --------------------------------------------------------------------
# Hash helpers
# --------------------------------------------------------------------
def _real_git_blob_sha1(content: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\x00" + content).hexdigest()


def test_git_blob_sha1_matches_manual(tmp_path):
    content = b"hello world\n"
    p = tmp_path / "f.txt"
    p.write_bytes(content)
    assert dl._git_blob_sha1(p) == _real_git_blob_sha1(content)


def test_git_blob_sha1_matches_git_hash_object(tmp_path):
    """Cross-check against the real ``git hash-object`` when git is available."""
    content = b"some dataset bytes \x00\x01\x02 spanning\n"
    p = tmp_path / "blob.bin"
    p.write_bytes(content)
    try:
        out = subprocess.run(["git", "hash-object", str(p)], capture_output=True, text=True, check=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    assert dl._git_blob_sha1(p) == out


def test_git_blob_sha1_is_not_sha256(tmp_path):
    """The bug was using sha256; make sure the two are actually different values."""
    content = b"x" * 1000
    p = tmp_path / "f"
    p.write_bytes(content)
    assert dl._git_blob_sha1(p) != hashlib.sha256(content).hexdigest()
    assert len(dl._git_blob_sha1(p)) == 40  # sha1 hex
    assert len(hashlib.sha256(content).hexdigest()) == 64


def test_sha256_file_streamed(tmp_path):
    content = b"lfs content" * 100000
    p = tmp_path / "big"
    p.write_bytes(content)
    assert dl._sha256_file(p) == hashlib.sha256(content).hexdigest()


# --------------------------------------------------------------------
# Regular file verification (git blob SHA-1)
# --------------------------------------------------------------------
def _patch_download(monkeypatch, content: bytes):
    """Make _download_file write `content` to dest, and disable network LFS info."""

    def fake_download_file(url, dest, token, desc="", file_size=0):
        Path(dest).write_bytes(content)
        return dest

    monkeypatch.setattr(dl, "_download_file", fake_download_file)


def test_regular_file_passes_git_blob_sha1(tmp_path, monkeypatch):
    content = b"regular file content\n"
    _patch_download(monkeypatch, content)
    # Not an LFS file:
    monkeypatch.setattr(dl, "_fetch_lfs_pointer_info", lambda *a, **k: (False, None, None))

    dest = tmp_path / "out.txt"
    dl._download_file_with_lfs(
        "user/repo",
        "out.txt",
        str(dest),
        token=None,
        dest_dir=tmp_path,
        expected_sha=_real_git_blob_sha1(content),
    )
    assert dest.exists()
    assert dest.read_bytes() == content


def test_regular_file_deleted_on_real_mismatch(tmp_path, monkeypatch):
    content = b"corrupted bytes"
    _patch_download(monkeypatch, content)
    monkeypatch.setattr(dl, "_fetch_lfs_pointer_info", lambda *a, **k: (False, None, None))

    dest = tmp_path / "out.txt"
    with pytest.raises(ValueError, match="Git blob SHA-1 mismatch"):
        dl._download_file_with_lfs(
            "user/repo",
            "out.txt",
            str(dest),
            token=None,
            dest_dir=tmp_path,
            expected_sha="0" * 40,  # wrong sha -> mismatch
        )
    assert not dest.exists()  # corrupt file removed


# --------------------------------------------------------------------
# LFS verification (oid / sha256 of real content)
# --------------------------------------------------------------------
def test_lfs_file_verified_against_oid_not_blob_sha(tmp_path, monkeypatch):
    """media returns real content; verify against the LFS oid, ignore blob sha."""
    real_content = b"LFS real content payload " * 5000
    oid = hashlib.sha256(real_content).hexdigest()
    _patch_download(monkeypatch, real_content)
    # Contents API says: this is an LFS file with this oid + real size.
    monkeypatch.setattr(dl, "_fetch_lfs_pointer_info", lambda *a, **k: (True, oid, len(real_content)))

    dest = tmp_path / "big.zip"
    # expected_sha is the git blob sha of the 134-byte pointer — must be ignored
    # for LFS content verification.
    dl._download_file_with_lfs(
        "user/repo",
        "big.zip",
        str(dest),
        token=None,
        dest_dir=tmp_path,
        expected_size=134,  # pointer size (wrong for content)
        expected_sha="deadbeef" * 5,  # pointer blob sha (must not be used)
    )
    assert dest.exists()
    assert dest.stat().st_size == len(real_content)


def test_lfs_file_deleted_on_oid_mismatch(tmp_path, monkeypatch):
    real_content = b"the actual bytes"
    _patch_download(monkeypatch, real_content)
    wrong_oid = "a" * 64
    monkeypatch.setattr(dl, "_fetch_lfs_pointer_info", lambda *a, **k: (True, wrong_oid, len(real_content)))

    dest = tmp_path / "big.zip"
    with pytest.raises(ValueError, match="LFS content"):
        dl._download_file_with_lfs(
            "user/repo", "big.zip", str(dest), token=None, dest_dir=tmp_path, expected_sha="x" * 40
        )
    assert not dest.exists()


def test_lfs_fast_skip_uses_real_size(tmp_path, monkeypatch):
    """A pre-existing file matching the real LFS size skips re-download."""
    real_content = b"already here" * 1000
    oid = hashlib.sha256(real_content).hexdigest()
    dest = tmp_path / "big.zip"
    dest.write_bytes(real_content)

    called = {"downloaded": False}

    def fake_download_file(url, d, token, desc="", file_size=0):
        called["downloaded"] = True
        Path(d).write_bytes(real_content)
        return d

    monkeypatch.setattr(dl, "_download_file", fake_download_file)
    monkeypatch.setattr(dl, "_fetch_lfs_pointer_info", lambda *a, **k: (True, oid, len(real_content)))

    dl._download_file_with_lfs(
        "user/repo",
        "big.zip",
        str(dest),
        token=None,
        dest_dir=tmp_path,
        expected_size=134,  # git/trees pointer size — must NOT drive the skip
        expected_sha="s" * 40,
    )
    # File already matched the real size, so no re-download happened.
    assert called["downloaded"] is False
    assert dest.read_bytes() == real_content
