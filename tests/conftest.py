"""pytest configuration for paddleOcean tests."""

import os
import sys

# Ensure the ocean package is importable from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Register the RunIf pytest_collection_modifyitems hook so env-based
# filtering (OCEAN_RUN_ONLY_CUDA_TESTS, OCEAN_RUN_STANDALONE_TESTS)
# works for multi-GPU test selection.
from tests.helpers.runif import pytest_collection_modifyitems  # noqa: F401, E402
