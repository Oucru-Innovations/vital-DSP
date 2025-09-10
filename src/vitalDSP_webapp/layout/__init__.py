"""
Layout module for vitalDSP webapp.

This module provides a clean, modular structure for all layout components
used in the Dash application.
"""

# Common components
from .common.header import Header
from .common.sidebar import Sidebar
from .common.footer import Footer

# Page layouts
from .pages.upload_page import upload_layout
from .pages.analysis_pages import (
    time_domain_layout,
    frequency_layout,
    filtering_layout,
    physiological_layout,
    respiratory_layout,
    features_layout,
    transforms_layout,
    quality_layout,
    advanced_layout,
    health_report_layout,
    settings_layout,
)

__all__ = [
    # Common components
    "Header",
    "Sidebar",
    "Footer",
    # Page layouts
    "upload_layout",
    "time_domain_layout",
    "frequency_layout",
    "filtering_layout",
    "physiological_layout",
    "respiratory_layout",
    "features_layout",
    "transforms_layout",
    "quality_layout",
    "advanced_layout",
    "health_report_layout",
    "settings_layout",
]
