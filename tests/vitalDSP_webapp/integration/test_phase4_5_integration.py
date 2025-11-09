"""
Test script for Phase 4 & 5: Display Enhancement and Navigation Updates

This script verifies that:
1. Display functions show entropy and fractal features
2. Display functions show preprocessing and advanced options info
3. Sidebar navigation updated
4. Route redirect from /features to /advanced works
"""

import sys
import io
import numpy as np
import inspect

# Fix encoding for Windows console (only when run directly, not under pytest)
