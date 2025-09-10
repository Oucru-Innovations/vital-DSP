"""Basic tests for quality_callbacks.py module."""

import pytest
import numpy as np
from unittest.mock import Mock

# Test data
SAMPLE_DATA = np.sin(2 * np.pi * np.linspace(0, 10, 1000)) + 0.1 * np.random.randn(1000)
SAMPLE_TIME = np.linspace(0, 10, 1000)
SAMPLE_FREQ = 100

try:
    from vitalDSP_webapp.callbacks.analysis.quality_callbacks import (
        register_quality_callbacks
    )
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

@pytest.mark.skipif(not AVAILABLE, reason="Module not available")
class TestBasicFunctions:
    def test_register_callbacks(self):
        mock_app = Mock()
        mock_app.callback = Mock(return_value=lambda f: f)
        register_quality_callbacks(mock_app)
        assert mock_app.callback.called
