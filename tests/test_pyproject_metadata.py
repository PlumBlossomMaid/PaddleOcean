"""Test pyproject.toml metadata for PyPI release readiness."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


@pytest.fixture
def pyproject():
    path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


class TestPyProjectMetadata:
    def test_name(self, pyproject):
        assert pyproject["project"]["name"] == "paddleocean"

    def test_version(self, pyproject):
        version = pyproject["project"]["version"]
        assert isinstance(version, str)
        parts = version.split(".")
        assert len(parts) >= 2, f"Version {version} should have at least major.minor"

    def test_description(self, pyproject):
        desc = pyproject["project"]["description"]
        assert "PaddlePaddle" in desc or "paddle" in desc.lower()

    def test_requires_python(self, pyproject):
        assert pyproject["project"]["requires-python"] == ">=3.9"

    def test_license(self, pyproject):
        assert "license" in pyproject["project"]
        assert "file" in pyproject["project"]["license"]

    def test_authors(self, pyproject):
        authors = pyproject["project"]["authors"]
        assert len(authors) >= 1
        assert "name" in authors[0]

    def test_keywords(self, pyproject):
        keywords = pyproject["project"]["keywords"]
        assert isinstance(keywords, list)
        assert len(keywords) >= 3
        assert "paddlepaddle" in keywords

    def test_classifiers(self, pyproject):
        classifiers = pyproject["project"]["classifiers"]
        assert isinstance(classifiers, list)
        assert len(classifiers) >= 5
        # Must have license classifier
        assert any("Apache" in c for c in classifiers)
        # Must have Python version classifiers
        assert any("Python :: 3.9" in c for c in classifiers)
        assert any("Python :: 3.10" in c for c in classifiers)

    def test_urls(self, pyproject):
        urls = pyproject["project"]["urls"]
        assert "Homepage" in urls or "Repository" in urls
        assert "github.com" in urls.get("Homepage", "") or "gitee.com" in urls.get("Homepage", "")

    def test_dependencies(self, pyproject):
        deps = pyproject["project"]["dependencies"]
        assert isinstance(deps, list)
        assert any("click" in d for d in deps)
        assert any("requests" in d for d in deps)

    def test_console_scripts(self, pyproject):
        scripts = pyproject["project"]["scripts"]
        assert "ocean" in scripts
        assert "ocean.cli" in scripts["ocean"]

    def test_optional_deps_dev(self, pyproject):
        dev = pyproject["project"]["optional-dependencies"]["dev"]
        assert any("pytest" in d for d in dev)
        assert any("ruff" in d for d in dev)

    def test_readme(self, pyproject):
        # readme should be specified
        assert "readme" in pyproject["project"]
        assert pyproject["project"]["readme"] == "README.md"

    def test_license_file_exists(self):
        path = Path(__file__).resolve().parent.parent / "LICENSE"
        assert path.exists()

    def test_readme_file_exists(self):
        path = Path(__file__).resolve().parent.parent / "README.md"
        assert path.exists()

    def test_ruff_config(self, pyproject):
        assert "tool" in pyproject
        assert "ruff" in pyproject["tool"]
        assert pyproject["tool"]["ruff"]["target-version"] == "py39"
        assert pyproject["tool"]["ruff"]["line-length"] == 120
