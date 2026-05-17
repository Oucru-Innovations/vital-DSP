"""
Features callbacks module for vitalDSP webapp.

This module contains callbacks for feature extraction and engineering.
"""

from .preview_callbacks import register_preview_callbacks

__all__ = [
    "register_preview_callbacks",
]
