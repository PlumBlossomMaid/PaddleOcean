"""Sphinx configuration for PaddleOcean."""

import os
import sys

# Add project root to path so autodoc can import ocean
sys.path.insert(0, os.path.abspath(".."))

# Suppress Paddle startup noise during doc build
os.environ["PADDLE_IGNORE_DEPRECATION"] = "1"

project = "PaddleOcean"
copyright = "2026, PlumBlossomMaid"
author = "PlumBlossomMaid"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # Google/NumPy style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx.ext.todo",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Theme
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = None

# autodoc settings
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
    "special-members": "__init__",
}
autodoc_typehints = "description"
autodoc_mock_imports = []

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# Master document
master_doc = "index"
