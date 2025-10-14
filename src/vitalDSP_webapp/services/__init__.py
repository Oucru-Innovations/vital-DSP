"""
Services module for vitalDSP webapp.

This module provides a clean, modular structure for all service components
used in the application.
"""

# Data services
from .data.data_service import get_data_service

# Settings services
try:
    from .settings_service import (
        SettingsService,
        load_settings,
        save_settings,
        get_default_settings,
        validate_settings,
        merge_settings
    )
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False

__all__ = [
    "get_data_service",
]

if SETTINGS_AVAILABLE:
    __all__.extend([
        "SettingsService",
        "load_settings", 
        "save_settings",
        "get_default_settings",
        "validate_settings",
        "merge_settings"
    ])
