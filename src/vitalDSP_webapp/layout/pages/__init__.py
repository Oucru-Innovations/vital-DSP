"""
Page layouts for vitalDSP webapp.

This module contains layouts for various application pages.
"""

from .upload_page import upload_layout
from .analysis_pages import (
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
    settings_layout
)

__all__ = [
    'upload_layout',
    'time_domain_layout',
    'frequency_layout',
    'filtering_layout',
    'physiological_layout',
    'respiratory_layout',
    'features_layout',
    'transforms_layout',
    'quality_layout',
    'advanced_layout',
    'health_report_layout',
    'settings_layout',
]
