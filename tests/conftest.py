"""pytest configuration for paddleOcean tests."""

import os
import sys

# Ensure the ocean package is importable from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
