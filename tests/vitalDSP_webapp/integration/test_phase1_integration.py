"""
Test script for Phase 1: Feature Extraction Unification

This script verifies that:
1. Imports from features_callbacks work correctly
2. extract_advanced_features accepts new parameters
3. perform_advanced_analysis accepts new parameters
"""

import sys
import numpy as np

# Fix encoding for Windows console (only when run directly, not under pytest)
import io
