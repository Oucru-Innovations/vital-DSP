"""
Callbacks module for vitalDSP webapp.

This module provides a clean, modular structure for all callback functions
used in the Dash application.
"""

# Core callbacks
from .core.app_callbacks import register_sidebar_callbacks
from .core.page_routing_callbacks import register_page_routing_callbacks
from .core.upload_callbacks import register_upload_callbacks

# Analysis callbacks
from .analysis.vitaldsp_callbacks import register_vitaldsp_callbacks
from .analysis.frequency_filtering_callbacks import (
    register_frequency_filtering_callbacks,
)
from .analysis.signal_filtering_callbacks import register_signal_filtering_callbacks
from .analysis.respiratory_callbacks import register_respiratory_callbacks

# Feature callbacks
from .features.physiological_callbacks import register_physiological_callbacks
from .features.features_callbacks import register_features_callbacks
from .features.preview_callbacks import register_preview_callbacks

__all__ = [
    # Core
    "register_sidebar_callbacks",
    "register_page_routing_callbacks",
    "register_upload_callbacks",
    # Analysis
    "register_vitaldsp_callbacks",
    "register_frequency_filtering_callbacks",
    "register_signal_filtering_callbacks",
    "register_respiratory_callbacks",
    # Features
    "register_physiological_callbacks",
    "register_features_callbacks",
    "register_preview_callbacks",
]
