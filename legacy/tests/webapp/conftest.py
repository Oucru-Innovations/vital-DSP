"""
Test configuration and fixtures for vitalDSP webapp tests.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path so tests can import vitalDSP_webapp modules
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Also add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))
