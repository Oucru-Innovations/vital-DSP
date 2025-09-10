"""
Services module for vitalDSP webapp.

This module provides a clean, modular structure for all service components
used in the application.
"""

# Data services
from .data.data_service import get_data_service

__all__ = [
    "get_data_service",
]
