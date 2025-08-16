"""
Custom exceptions for the PPG analysis tool.

This module defines custom exception classes for different types of errors
that can occur during PPG data processing and analysis.
"""

from typing import Any, Dict, Optional, Union


class PPGError(Exception):
    """Base exception class for PPG analysis errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class FileError(PPGError):
    """Exception raised for file-related errors."""

    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        super().__init__(message, kwargs)
        self.file_path = file_path


class DataError(PPGError):
    """Exception raised for data processing errors."""

    def __init__(self, message: str, data_info: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(message, kwargs)
        self.data_info = data_info or {}


class ValidationError(PPGError):
    """Exception raised for data validation errors."""

    def __init__(self, message: str, field: Optional[str] = None, value: Any = None, **kwargs):
        super().__init__(message, kwargs)
        self.field = field
        self.value = value


class ProcessingError(PPGError):
    """Exception raised for signal processing errors."""

    def __init__(self, message: str, processing_step: Optional[str] = None, **kwargs):
        super().__init__(message, kwargs)
        self.processing_step = processing_step


class ConfigurationError(PPGError):
    """Exception raised for configuration errors."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(message, kwargs)
        self.config_key = config_key


# Specific error types
class FileNotFoundError(FileError):
    """Exception raised when a file is not found."""

    pass


class FileTooLargeError(FileError):
    """Exception raised when a file exceeds size limits."""

    pass


class InvalidFileFormatError(FileError):
    """Exception raised when a file has an invalid format."""

    pass


class InvalidColumnMappingError(DataError):
    """Exception raised when column mapping is invalid."""

    pass


class InsufficientDataError(DataError):
    """Exception raised when there is insufficient data for processing."""

    pass


class InvalidParameterError(ValidationError):
    """Exception raised when a parameter value is invalid."""

    pass


class SignalProcessingError(ProcessingError):
    """Exception raised when signal processing fails."""

    pass


class PeakDetectionError(ProcessingError):
    """Exception raised when peak detection fails."""

    pass


class HeartRateCalculationError(ProcessingError):
    """Exception raised when heart rate calculation fails."""

    pass


class SpO2CalculationError(ProcessingError):
    """Exception raised when SpO2 calculation fails."""

    pass
