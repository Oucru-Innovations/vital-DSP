"""
Logging configuration for the PPG analysis tool.

This module provides centralized logging configuration with different log levels
for development and production environments, using colorlog for enhanced console output.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

try:
    import colorlog

    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False

from .settings import settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_colors: bool = True,
) -> None:
    """
    Set up logging configuration for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        enable_colors: Whether to enable colored console output
    """
    # Determine log level
    if log_level is None:
        log_level = "DEBUG" if settings.debug else "INFO"

    # Convert string to logging level
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    simple_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler with colorlog (if available)
    if enable_colors and COLORLOG_AVAILABLE:
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
            style="%",
        )
        console_handler.setFormatter(console_formatter)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(simple_formatter)

    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)

    # Set specific logger levels
    logging.getLogger("dash").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Log the configuration
    logging.info(f"Logging configured with level: {log_level}")
    if log_file:
        logging.info(f"Log file: {log_file}")
    if enable_colors and COLORLOG_AVAILABLE:
        logging.info("Colored console output enabled")
    elif enable_colors and not COLORLOG_AVAILABLE:
        logging.info("Colored console output requested but colorlog not available")
    else:
        logging.info("Colored console output disabled")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_function_call(func_name: str, args: tuple = None, kwargs: dict = None) -> None:
    """
    Log function calls for debugging purposes.

    Args:
        func_name: Name of the function being called
        args: Function arguments
        kwargs: Function keyword arguments
    """
    if settings.debug:
        logger = get_logger(__name__)
        arg_str = f"args={args}" if args else ""
        kwarg_str = f"kwargs={kwargs}" if kwargs else ""
        params = ", ".join(filter(None, [arg_str, kwarg_str]))
        logger.debug(f"Calling {func_name}({params})")


def log_function_result(func_name: str, result: any = None, error: Exception = None) -> None:
    """
    Log function results or errors for debugging purposes.

    Args:
        func_name: Name of the function that was called
        result: Function return value
        error: Exception that occurred (if any)
    """
    if settings.debug:
        logger = get_logger(__name__)
        if error:
            logger.error(f"{func_name} failed with error: {error}")
        else:
            logger.debug(f"{func_name} completed successfully")


def log_computation_start(operation: str, **kwargs) -> None:
    """
    Log the start of a computation operation.

    Args:
        operation: Description of the operation
        **kwargs: Additional context parameters
    """
    logger = get_logger(__name__)
    context = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"🔄 Starting {operation} - {context}")


def log_computation_progress(operation: str, progress: str, **kwargs) -> None:
    """
    Log progress updates during computation.

    Args:
        operation: Description of the operation
        progress: Progress description
        **kwargs: Additional context parameters
    """
    logger = get_logger(__name__)
    context = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.debug(f"⏳ {operation}: {progress} - {context}")


def log_computation_complete(operation: str, result_summary: str = None, **kwargs) -> None:
    """
    Log the completion of a computation operation.

    Args:
        operation: Description of the operation
        result_summary: Summary of results
        **kwargs: Additional context parameters
    """
    logger = get_logger(__name__)
    context = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    if result_summary:
        logger.info(f"✅ {operation} completed - {result_summary} - {context}")
    else:
        logger.info(f"✅ {operation} completed - {context}")


def log_data_validation(data_type: str, data_shape: tuple = None, **kwargs) -> None:
    """
    Log data validation information.

    Args:
        data_type: Type of data being validated
        data_shape: Shape/dimensions of the data
        **kwargs: Additional validation parameters
    """
    logger = get_logger(__name__)
    context = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    shape_info = f" (shape: {data_shape})" if data_shape else ""
    logger.debug(f"🔍 Validating {data_type}{shape_info} - {context}")


def log_analysis_step(step_name: str, input_info: str = None, **kwargs) -> None:
    """
    Log analysis step information.

    Args:
        step_name: Name of the analysis step
        input_info: Information about input data
        **kwargs: Additional step parameters
    """
    logger = get_logger(__name__)
    context = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    input_str = f" - Input: {input_info}" if input_info else ""
    logger.debug(f"📊 Analysis step: {step_name}{input_str} - {context}")


# Default logging setup
def setup_default_logging() -> None:
    """Set up default logging configuration."""
    try:
        # Create logs directory if it doesn't exist
        log_file = Path("logs/app.log")
        log_file.parent.mkdir(exist_ok=True)
        setup_logging(log_file=str(log_file), enable_colors=True)
    except Exception as e:
        # Fallback to console-only logging if file logging fails
        print(f"Warning: File logging setup failed, using console-only logging: {e}")
        setup_logging(enable_colors=True)


# Initialize logging when module is imported (only if no handlers exist)
if not logging.getLogger().handlers:
    try:
        setup_default_logging()
    except Exception as e:
        # If setup fails, just use basic console logging
        print(f"Warning: Logging setup failed: {e}")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
