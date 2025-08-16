"""
Error handling utilities for vitalDSP webapp.
Provides consistent error handling and user feedback.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Union
from dash import html
import dash_bootstrap_components as dbc

from ..config.settings import ui_styles

logger = logging.getLogger(__name__)


class WebappError(Exception):
    """Base exception class for webapp errors."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class DataProcessingError(WebappError):
    """Exception raised for data processing errors."""
    pass


class FileUploadError(WebappError):
    """Exception raised for file upload errors."""
    pass


class ValidationError(WebappError):
    """Exception raised for validation errors."""
    pass


class AnalysisError(WebappError):
    """Exception raised for analysis errors."""
    pass


def handle_error(error: Exception, context: str = "Unknown") -> Dict[str, Any]:
    """
    Handle an error and return standardized error information.
    
    Parameters
    ----------
    error : Exception
        The error that occurred
    context : str
        Context where the error occurred
        
    Returns
    -------
    Dict[str, Any]
        Standardized error information
    """
    error_info = {
        'type': type(error).__name__,
        'message': str(error),
        'context': context,
        'timestamp': None,
        'traceback': None
    }
    
    # Add timestamp
    from datetime import datetime
    error_info['timestamp'] = datetime.now().isoformat()
    
    # Add traceback for logging
    if logger.isEnabledFor(logging.DEBUG):
        error_info['traceback'] = traceback.format_exc()
    
    # Log the error
    logger.error(f"Error in {context}: {error}", exc_info=True)
    
    return error_info


def create_error_alert(error: Union[Exception, str], 
                      title: str = "Error",
                      color: str = "danger",
                      show_details: bool = False) -> dbc.Alert:
    """
    Create a standardized error alert component.
    
    Parameters
    ----------
    error : Union[Exception, str]
        Error object or error message string
    title : str
        Title for the error alert
    color : str
        Bootstrap color for the alert
    show_details : bool
        Whether to show error details
        
    Returns
    -------
    dbc.Alert
        Error alert component
    """
    if isinstance(error, Exception):
        message = str(error)
        error_type = type(error).__name__
    else:
        message = error
        error_type = "Error"
    
    alert_content = [
        html.H5(f"❌ {title}", className="alert-heading"),
        html.P(message)
    ]
    
    if show_details and isinstance(error, Exception):
        alert_content.extend([
            html.Hr(),
            html.P([
                html.Strong("Error Type: "),
                error_type
            ], className="mb-1"),
            html.P([
                html.Strong("Context: "),
                getattr(error, 'context', 'Unknown')
            ], className="mb-1")
        ])
    
    return dbc.Alert(
        alert_content,
        color=color,
        className="mt-3"
    )


def create_warning_alert(message: str, title: str = "Warning") -> dbc.Alert:
    """
    Create a standardized warning alert component.
    
    Parameters
    ----------
    message : str
        Warning message
    title : str
        Title for the warning alert
        
    Returns
    -------
    dbc.Alert
        Warning alert component
    """
    return dbc.Alert([
        html.H5(f"⚠️ {title}", className="alert-heading"),
        html.P(message)
    ], color="warning", className="mt-3")


def create_info_alert(message: str, title: str = "Information") -> dbc.Alert:
    """
    Create a standardized info alert component.
    
    Parameters
    ----------
    message : str
        Information message
    title : str
        Title for the info alert
        
    Returns
    -------
    dbc.Alert
        Info alert component
    """
    return dbc.Alert([
        html.H5(f"ℹ️ {title}", className="alert-heading"),
        html.P(message)
    ], color="info", className="mt-3")


def create_success_alert(message: str, title: str = "Success") -> dbc.Alert:
    """
    Create a standardized success alert component.
    
    Parameters
    ----------
    message : str
        Success message
    title : str
        Title for the success alert
        
    Returns
    -------
    dbc.Alert
        Success alert component
    """
    return dbc.Alert([
        html.H5(f"✅ {title}", className="alert-heading"),
        html.P(message)
    ], color="success", className="mt-3")


def safe_execute(func, *args, default_return=None, error_context: str = "Unknown", **kwargs):
    """
    Safely execute a function with error handling.
    
    Parameters
    ----------
    func : callable
        Function to execute
    *args : tuple
        Positional arguments for the function
    default_return : Any
        Default return value if function fails
    error_context : str
        Context for error logging
    **kwargs : dict
        Keyword arguments for the function
        
    Returns
    -------
    Any
        Function result or default_return if function fails
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handle_error(e, error_context)
        return default_return


def validate_required_fields(data: Dict[str, Any], required_fields: list, context: str = "Data validation") -> None:
    """
    Validate that required fields are present in data.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Data to validate
    required_fields : list
        List of required field names
    context : str
        Context for error messages
        
    Raises
    ------
    ValidationError
        If required fields are missing
    """
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
    
    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}",
            error_code="MISSING_FIELDS",
            details={'missing_fields': missing_fields, 'context': context}
        )


def validate_data_types(data: Dict[str, Any], field_types: Dict[str, type], context: str = "Data type validation") -> None:
    """
    Validate data types of fields.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Data to validate
    field_types : Dict[str, type]
        Dictionary mapping field names to expected types
    context : str
        Context for error messages
        
    Raises
    ------
    ValidationError
        If data types don't match expected types
    """
    type_errors = []
    
    for field, expected_type in field_types.items():
        if field in data and data[field] is not None:
            if not isinstance(data[field], expected_type):
                type_errors.append(f"{field}: expected {expected_type.__name__}, got {type(data[field]).__name__}")
    
    if type_errors:
        raise ValidationError(
            f"Data type validation failed: {'; '.join(type_errors)}",
            error_code="TYPE_MISMATCH",
            details={'type_errors': type_errors, 'context': context}
        )


def format_error_message(error: Exception, include_traceback: bool = False) -> str:
    """
    Format an error message for display.
    
    Parameters
    ----------
    error : Exception
        The error to format
    include_traceback : bool
        Whether to include traceback information
        
    Returns
    -------
    str
        Formatted error message
    """
    if isinstance(error, WebappError):
        message = str(error)
        if error.details:
            message += f"\nDetails: {error.details}"
    else:
        message = str(error)
    
    if include_traceback:
        message += f"\n\nTraceback:\n{traceback.format_exc()}"
    
    return message
