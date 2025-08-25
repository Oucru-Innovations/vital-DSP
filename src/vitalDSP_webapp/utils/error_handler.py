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
        
        # Add timestamp
        from datetime import datetime
        self.timestamp = datetime.now()
        
        # Add context attribute for backward compatibility
        self.context = details or {}
        
        # Add error_type attribute for backward compatibility
        self.error_type = error_code
    
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
                      show_details: bool = False,
                      dismissible: bool = False) -> dbc.Alert:
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
        className="mt-3",
        dismissable=dismissible  # DBC uses 'dismissable' not 'dismissible'
    )


def handle_upload_error(error: Exception, context: dict = None) -> dict:
    """
    Handle file upload errors specifically.
    
    Parameters
    ----------
    error : Exception
        The upload error that occurred
    context : dict, optional
        Additional context about the upload
        
    Returns
    -------
    dict
        Standardized error response for upload errors
    """
    error_info = handle_error(error, "File Upload")
    
    # Add upload-specific error handling
    if isinstance(error, FileUploadError):
        error_info['upload_context'] = context or {}
        error_info['suggestions'] = get_upload_error_suggestions(error)
    
    # Add the expected keys for the test
    error_info['error'] = True
    error_info['message'] = str(error)
    
    return error_info


def handle_processing_error(error: Exception, context: dict = None) -> dict:
    """
    Handle data processing errors specifically.
    
    Parameters
    ----------
    error : Exception
        The processing error that occurred
    context : dict, optional
        Additional context about the processing
        
    Returns
    -------
    dict
        Standardized error response for processing errors
    """
    error_info = handle_error(error, "Data Processing")
    
    # Add processing-specific error handling
    if isinstance(error, DataProcessingError):
        error_info['processing_context'] = context or {}
        error_info['suggestions'] = get_processing_error_suggestions(error)
    
    # Add the expected keys for the test
    error_info['error'] = True
    error_info['message'] = str(error)
    
    return error_info


def get_processing_error_suggestions(error: Exception) -> list:
    """Get user-friendly suggestions for processing errors."""
    suggestions = []
    
    if "data format" in str(error).lower():
        suggestions.append("Check if the data format is supported")
        suggestions.append("Ensure the data has the expected structure")
    elif "column mapping" in str(error).lower():
        suggestions.append("Verify the column mapping configuration")
        suggestions.append("Check if all required columns are present")
    elif "sampling frequency" in str(error).lower():
        suggestions.append("Verify the sampling frequency value")
        suggestions.append("Ensure the sampling frequency is positive")
    else:
        suggestions.append("Check the data for any anomalies")
        suggestions.append("Verify the processing parameters")
    
    return suggestions


def handle_analysis_error(error: Exception, context: dict = None) -> dict:
    """
    Handle analysis errors specifically.
    
    Parameters
    ----------
    error : Exception
        The analysis error that occurred
    context : dict, optional
        Additional context about the analysis
        
    Returns
    -------
    dict
        Standardized error response for analysis errors
    """
    error_info = handle_error(error, "Data Analysis")
    
    # Add analysis-specific error handling
    if isinstance(error, AnalysisError):
        error_info['analysis_context'] = context or {}
        error_info['suggestions'] = get_analysis_error_suggestions(error)
    
    # Add the expected keys for the test
    error_info['error'] = True
    error_info['message'] = str(error)
    
    return error_info


def get_analysis_error_suggestions(error: Exception) -> list:
    """Get user-friendly suggestions for analysis errors."""
    suggestions = []
    
    if "insufficient data" in str(error).lower():
        suggestions.append("Ensure you have enough data points for analysis")
        suggestions.append("Check if the data window size is appropriate")
    elif "parameter" in str(error).lower():
        suggestions.append("Verify the analysis parameters")
        suggestions.append("Check if parameter values are within valid ranges")
    else:
        suggestions.append("Review the analysis configuration")
        suggestions.append("Check if the data quality is sufficient")
    
    return suggestions


def log_error_with_context(error: Exception, context: dict, level: str = "ERROR"):
    """
    Log an error with additional context information.
    
    Parameters
    ----------
    error : Exception
        The error to log
    context : dict
        Additional context information
    level : str
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger = logging.getLogger(__name__)
    
    context_str = " | ".join([f"{k}: {v}" for k, v in context.items()])
    error_msg = f"{str(error)} | Context: {context_str}"
    
    if level.upper() == "DEBUG":
        logger.debug(error_msg)
    elif level.upper() == "INFO":
        logger.info(error_msg)
    elif level.upper() == "WARNING":
        logger.warning(error_msg)
    elif level.upper() == "ERROR":
        logger.error(error_msg)
    elif level.upper() == "CRITICAL":
        logger.critical(error_msg)
    else:
        logger.error(error_msg)


def create_user_friendly_error_message(error: Exception) -> str:
    """
    Create a user-friendly error message from an exception.
    
    Parameters
    ----------
    error : Exception
        The error to convert to a user-friendly message
        
    Returns
    -------
    str
        User-friendly error message
    """
    if isinstance(error, WebappError):
        error_type = error.error_code or "Unknown Error"
        if "upload" in error_type.lower():
            return f"Upload Error: {error.message}. Please check your file and try again."
        elif "processing" in error_type.lower():
            return f"Processing Error: {error.message}. Please verify your data and settings."
        elif "analysis" in error_type.lower():
            return f"Analysis Error: {error.message}. Please check your analysis parameters."
        else:
            return f"{error_type}: {error.message}"
    else:
        # Generic error handling
        error_msg = str(error)
        if "file" in error_msg.lower():
            return f"File Error: {error_msg}. Please check your file and try again."
        elif "data" in error_msg.lower():
            return f"Data Error: {error_msg}. Please verify your data and try again."
        else:
            return f"An error occurred: {error_msg}. Please try again or contact support."


def format_error_for_display(error: Exception, include_details: bool = False) -> dict:
    """
    Format an error for display in the UI.
    
    Parameters
    ----------
    error : Exception
        The error to format
    include_details : bool
        Whether to include detailed error information
        
    Returns
    -------
    dict
        Formatted error information for display
    """
    from datetime import datetime
    
    formatted_error = {
        "message": str(error),
        "type": getattr(error, 'error_code', type(error).__name__),
        "timestamp": datetime.now().isoformat()
    }
    
    # Add context if available
    if hasattr(error, 'details') and error.details:
        formatted_error["context"] = error.details
    
    # Add traceback if requested and available
    if include_details:
        import traceback
        formatted_error["traceback"] = traceback.format_exc()
    
    return formatted_error


def get_upload_error_suggestions(error: Exception) -> list:
    """Get user-friendly suggestions for upload errors."""
    suggestions = []
    
    if "file size" in str(error).lower():
        suggestions.append("Try uploading a smaller file")
        suggestions.append("Check if the file is compressed")
    elif "file format" in str(error).lower():
        suggestions.append("Ensure the file is in CSV, Excel, or text format")
        suggestions.append("Check if the file extension matches the content")
    elif "permission" in str(error).lower():
        suggestions.append("Check file permissions")
        suggestions.append("Try running the application with appropriate privileges")
    else:
        suggestions.append("Check the file for corruption")
        suggestions.append("Try uploading a different file")
    
    return suggestions


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
