"""
Analysis callbacks module for vitalDSP webapp.

This module contains callbacks for signal analysis and processing.
"""

from .vitaldsp_callbacks import register_vitaldsp_callbacks
from .frequency_filtering_callbacks import register_frequency_filtering_callbacks
from .respiratory_callbacks import register_respiratory_callbacks
from .quality_callbacks import register_quality_callbacks

__all__ = [
    "register_vitaldsp_callbacks",
    "register_frequency_filtering_callbacks",
    "register_respiratory_callbacks",
    "register_quality_callbacks",
]
