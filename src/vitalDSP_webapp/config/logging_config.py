"""
Logging configuration for vitalDSP webapp.
Provides consistent logging setup across all components.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

from .settings import app_config


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Setup logging configuration for the webapp.
    
    Parameters
    ----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : Optional[str]
        Path to log file. If None, logs only to console
    log_format : str
        Log message format string
    max_bytes : int
        Maximum size of log file before rotation
    backup_count : int
        Number of backup log files to keep
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Default log format
    if log_format is None:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(funcName)s:%(lineno)d - %(message)s"
        )
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if log_file is specified)
    if log_file:
        # Ensure log directory exists
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger('dash').setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    
    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {log_level}, File: {log_file or 'Console only'}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Parameters
    ----------
    name : str
        Logger name (usually __name__)
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    return logging.getLogger(name)


def log_function_call(func_name: str, args: tuple = None, kwargs: dict = None, logger: logging.Logger = None):
    """
    Decorator to log function calls with parameters.
    
    Parameters
    ----------
    func_name : str
        Name of the function being called
    args : tuple
        Positional arguments
    kwargs : dict
        Keyword arguments
    logger : logging.Logger
        Logger instance to use
    """
    if logger is None:
        logger = get_logger(__name__)
    
    args_str = str(args) if args else "()"
    kwargs_str = str(kwargs) if kwargs else "{}"
    
    logger.debug(f"Function call: {func_name}{args_str}, kwargs={kwargs_str}")


def log_performance(func_name: str, execution_time: float, logger: logging.Logger = None):
    """
    Log function performance metrics.
    
    Parameters
    ----------
    func_name : str
        Name of the function
    execution_time : float
        Execution time in seconds
    logger : logging.Logger
        Logger instance to use
    """
    if logger is None:
        logger = get_logger(__name__)
    
    if execution_time > 1.0:
        logger.warning(f"Slow function execution: {func_name} took {execution_time:.2f}s")
    elif execution_time > 0.1:
        logger.info(f"Function execution: {func_name} took {execution_time:.3f}s")
    else:
        logger.debug(f"Function execution: {func_name} took {execution_time:.3f}s")


def log_data_operation(operation: str, data_size: int, data_type: str = "unknown", logger: logging.Logger = None):
    """
    Log data operation metrics.
    
    Parameters
    ----------
    operation : str
        Type of operation (upload, process, analyze, etc.)
    data_size : int
        Size of data in bytes or number of records
    data_type : str
        Type of data being processed
    logger : logging.Logger
        Logger instance to use
    """
    if logger is None:
        logger = get_logger(__name__)
    
    if data_size > 100 * 1024 * 1024:  # 100MB
        size_str = f"{data_size / (1024 * 1024):.1f}MB"
    elif data_size > 1024 * 1024:  # 1MB
        size_str = f"{data_size / (1024 * 1024):.2f}MB"
    elif data_size > 1024:  # 1KB
        size_str = f"{data_size / 1024:.1f}KB"
    else:
        size_str = f"{data_size}B"
    
    logger.info(f"Data operation: {operation} - {data_type} ({size_str})")


def log_error_with_context(error: Exception, context: str, logger: logging.Logger = None, include_traceback: bool = True):
    """
    Log an error with context information.
    
    Parameters
    ----------
    error : Exception
        The error that occurred
    context : str
        Context where the error occurred
    logger : logging.Logger
        Logger instance to use
    include_traceback : bool
        Whether to include traceback information
    """
    if logger is None:
        logger = get_logger(__name__)
    
    error_msg = f"Error in {context}: {str(error)}"
    
    if include_traceback:
        logger.error(error_msg, exc_info=True)
    else:
        logger.error(error_msg)


def setup_development_logging() -> None:
    """Setup logging for development environment."""
    setup_logging(
        log_level="DEBUG",
        log_file=os.path.join(app_config.UPLOAD_FOLDER, "vitaldsp_webapp_dev.log"),
        max_bytes=5 * 1024 * 1024,  # 5MB
        backup_count=3
    )


def setup_production_logging() -> None:
    """Setup logging for production environment."""
    setup_logging(
        log_level="INFO",
        log_file=os.path.join(app_config.UPLOAD_FOLDER, "vitaldsp_webapp.log"),
        max_bytes=50 * 1024 * 1024,  # 50MB
        backup_count=10
    )


def setup_test_logging() -> None:
    """Setup logging for testing environment."""
    setup_logging(
        log_level="WARNING",
        log_file=None  # Console only for tests
    )


# Auto-setup logging based on environment
if app_config.DEBUG:
    setup_development_logging()
else:
    setup_production_logging()
