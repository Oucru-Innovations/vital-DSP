"""
Utilities module for vitalDSP webapp.
"""

from .data_processor import DataProcessor
from .error_handler import (
    WebappError,
    DataProcessingError,
    FileUploadError,
    ValidationError,
    AnalysisError,
    handle_error,
    create_error_alert,
    create_warning_alert,
    create_info_alert,
    create_success_alert,
    safe_execute,
    validate_required_fields,
    validate_data_types,
    format_error_message
)

__all__ = [
    'DataProcessor',
    'WebappError',
    'DataProcessingError',
    'FileUploadError',
    'ValidationError',
    'AnalysisError',
    'handle_error',
    'create_error_alert',
    'create_warning_alert',
    'create_info_alert',
    'create_success_alert',
    'safe_execute',
    'validate_required_fields',
    'validate_data_types',
    'format_error_message'
]
