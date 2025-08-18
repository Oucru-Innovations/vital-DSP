"""
Core callbacks module for vitalDSP webapp.

This module contains the essential callbacks for basic application functionality.
"""

from .app_callbacks import register_sidebar_callbacks
from .page_routing_callbacks import register_page_routing_callbacks
from .upload_callbacks import register_upload_callbacks

__all__ = [
    'register_sidebar_callbacks',
    'register_page_routing_callbacks',
    'register_upload_callbacks',
]
