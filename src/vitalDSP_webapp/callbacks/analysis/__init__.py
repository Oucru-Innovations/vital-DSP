"""
Analysis callbacks module for vitalDSP webapp.

This module contains callbacks for signal analysis and processing.
"""

from .vitaldsp_callbacks import register_vitaldsp_callbacks
from .time_domain_callbacks import register_time_domain_callbacks
from .frequency_filtering_callbacks import register_frequency_filtering_callbacks

__all__ = [
    "register_vitaldsp_callbacks",
    "register_time_domain_callbacks",
    "register_frequency_filtering_callbacks",
]
