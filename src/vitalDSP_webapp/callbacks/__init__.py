"""
Callbacks module for vitalDSP webapp.

This module provides a clean, modular structure for all callback functions
used in the Dash application.
"""

from .core.header_monitoring_callbacks import register_header_monitoring_callbacks

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
from .analysis.quality_callbacks import register_quality_callbacks
from .analysis.advanced_callbacks import register_advanced_callbacks
from .analysis.health_report_callbacks import register_health_report_callbacks
from .analysis.settings_callbacks import register_settings_callbacks
from .analysis.tasks_callbacks import register_tasks_callbacks
from .analysis.pipeline_callbacks import register_pipeline_callbacks

# Feature callbacks
from .features.physiological_callbacks import register_physiological_callbacks
from .features.features_callbacks import register_features_callbacks
from .features.preview_callbacks import register_preview_callbacks

__all__ = [
    # Core
    "register_sidebar_callbacks",
    "register_page_routing_callbacks",
    "register_upload_callbacks",
    "register_header_monitoring_callbacks",
    # Analysis
    "register_vitaldsp_callbacks",
    "register_frequency_filtering_callbacks",
    "register_signal_filtering_callbacks",
    "register_respiratory_callbacks",
    "register_quality_callbacks",
    "register_advanced_callbacks",
    "register_health_report_callbacks",
    "register_settings_callbacks",
    "register_tasks_callbacks",
    "register_pipeline_callbacks",
    # Features
    "register_physiological_callbacks",
    "register_features_callbacks",
    "register_preview_callbacks",
]
