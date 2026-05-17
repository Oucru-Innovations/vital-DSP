"""
Page layouts for vitalDSP webapp.

This module contains layouts for various application pages.
"""

from .upload_page import upload_layout
from .time_domain_page import time_domain_layout
from .frequency_page import frequency_layout
from .filtering_page import filtering_layout

__all__ = [
    "upload_layout",
    "time_domain_layout",
    "frequency_layout",
    "filtering_layout",
]
