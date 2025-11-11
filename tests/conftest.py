"""
Test configuration and fixtures for vital-DSP tests.
"""

import os
import sys
from pathlib import Path
import pytest

# Get the project root directory (parent of tests directory)
project_root = Path(__file__).parent.parent
src_path = project_root / "src"

# Add the src directory to the Python path so tests can import modules
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Also add the current directory to the path
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Ensure we can import vitalDSP_webapp modules
try:
    import vitalDSP_webapp
except ImportError:
    # If direct import fails, try to add the parent directory
    parent_dir = str(project_root)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)


# Configure pytest-xdist to run serial tests in the same worker
def pytest_collection_modifyitems(items):
    """Mark tests with serial marker to run in the same xdist group."""
    for item in items:
        if 'serial' in item.keywords:
            item.add_marker(pytest.mark.xdist_group(name="serial"))
