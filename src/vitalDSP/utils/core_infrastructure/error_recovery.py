"""
Robust Error Handling and Recovery System for Large Data Processing

This module implements comprehensive error handling, recovery mechanisms,
and user-friendly error reporting for the large data processing pipeline.

Author: vitalDSP Development Team
Date: October 12, 2025
Version: 1.0.0
"""

"""
Utility Functions Module for Physiological Signal Processing

This module provides comprehensive capabilities for physiological
signal processing including ECG, PPG, EEG, and other vital signs.

Author: vitalDSP Team
Date: 2025-01-27
Version: 1.0.0

Key Features:
- Object-oriented design with comprehensive classes
- Multiple processing methods and functions
- NumPy integration for numerical computations
- SciPy integration for advanced signal processing
- Parallel processing capabilities

Examples:
--------
Basic usage:
    >>> import numpy as np
    >>> from vitalDSP.core_infrastructure.error_recovery import ErrorRecovery
    >>> signal = np.random.randn(1000)
    >>> processor = ErrorRecovery(signal)
    >>> result = processor.process()
    >>> print(f'Processing result: {result}')
"""


import os
import sys
import traceback
import logging
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
from datetime import datetime
from pathlib import Path
import functools
import warnings

from ..config_utilities.dynamic_config import DynamicConfigManager

# Configure logging
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Enumeration of error severity levels."""

    LOW = "low"  # Minor issues, processing can continue
    MEDIUM = "medium"  # Significant issues, may affect results
    HIGH = "high"  # Critical issues, processing may fail
    CRITICAL = "critical"  # Fatal errors, processing must stop


class ErrorCategory(Enum):
    """Enumeration of error categories."""

    MEMORY = "memory"  # Memory-related errors
    PROCESSING = "processing"  # Processing errors
    DATA = "data"  # Data-related errors
    CONFIGURATION = "configuration"  # Configuration errors
    SYSTEM = "system"  # System-level errors
    USER = "user"  # User input errors
    NETWORK = "network"  # Network-related errors
    FILE = "file"  # File I/O errors


@dataclass
class ErrorInfo:
    """Data class for error information."""

    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    error_type: str
    message: str
    details: Dict[str, Any]
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_method: Optional[str] = None


@dataclass
class RecoveryResult:
    """Data class for recovery result."""

    success: bool
    recovered_data: Optional[Any] = None
    recovery_method: Optional[str] = None
    warning_message: Optional[str] = None
    partial_results: Optional[Dict[str, Any]] = None


class ErrorRecoveryManager:
    """
    Manages error recovery strategies and partial result preservation.
    """

    def __init__(self, config_manager: Optional[DynamicConfigManager] = None):
        """
        Initialize error recovery manager.

        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager or DynamicConfigManager()
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.partial_results = {}
        self.recovery_history = []

    def _initialize_recovery_strategies(self) -> Dict[str, Callable]:
        """Initialize recovery strategies for different error types."""
        return {
            "memory_error": self._recover_from_memory_error,
            "data_corruption": self._recover_from_data_corruption,
            "processing_timeout": self._recover_from_timeout,
            "file_not_found": self._recover_from_file_error,
            "configuration_error": self._recover_from_config_error,
            "network_error": self._recover_from_network_error,
            "generic_error": self._recover_from_generic_error,
        }

    def attempt_recovery(
        self, error_info: ErrorInfo, context: Dict[str, Any]
    ) -> RecoveryResult:
        """
        Attempt to recover from an error.

        Args:
            error_info: Error information
            context: Processing context

        Returns:
            Recovery result
        """
        logger.info(f"Attempting recovery for error: {error_info.error_id}")

        # Determine recovery strategy
        recovery_strategy = self._determine_recovery_strategy(error_info)

        if recovery_strategy is None:
            logger.warning(
                f"No recovery strategy available for error: {error_info.error_type}"
            )
            return RecoveryResult(success=False)

        try:
            # Attempt recovery
            recovery_result = recovery_strategy(error_info, context)

            # Record recovery attempt
            self.recovery_history.append(
                {
                    "timestamp": datetime.now(),
                    "error_id": error_info.error_id,
                    "strategy": recovery_strategy.__name__,
                    "success": recovery_result.success,
                }
            )

            if recovery_result.success:
                logger.info(f"Recovery successful for error: {error_info.error_id}")
            else:
                logger.warning(f"Recovery failed for error: {error_info.error_id}")

            return recovery_result

        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            return RecoveryResult(success=False)

    def _determine_recovery_strategy(self, error_info: ErrorInfo) -> Optional[Callable]:
        """Determine appropriate recovery strategy for error."""
        error_type = error_info.error_type.lower()

        # Map error types to recovery strategies
        if "memory" in error_type or "out of memory" in error_info.message.lower():
            return self.recovery_strategies["memory_error"]
        elif "corrupt" in error_type or "invalid" in error_type:
            return self.recovery_strategies["data_corruption"]
        elif "timeout" in error_type or "timed out" in error_info.message.lower():
            return self.recovery_strategies["processing_timeout"]
        elif "file" in error_type or "not found" in error_info.message.lower():
            return self.recovery_strategies["file_not_found"]
        elif "config" in error_type or "configuration" in error_type:
            return self.recovery_strategies["configuration_error"]
        elif "network" in error_type or "connection" in error_type:
            return self.recovery_strategies["network_error"]
        else:
            return self.recovery_strategies["generic_error"]

    def _recover_from_memory_error(
        self, error_info: ErrorInfo, context: Dict[str, Any]
    ) -> RecoveryResult:
        """Recover from memory-related errors."""
        logger.info("Attempting memory error recovery")

        try:
            # Force garbage collection
            import gc

            gc.collect()

            # Reduce chunk size
            if "chunk_size" in context:
                original_chunk_size = context["chunk_size"]
                new_chunk_size = max(original_chunk_size // 2, 1000)
                context["chunk_size"] = new_chunk_size

                logger.info(
                    f"Reduced chunk size from {original_chunk_size} to {new_chunk_size}"
                )

                return RecoveryResult(
                    success=True,
                    recovery_method="chunk_size_reduction",
                    warning_message=f"Reduced chunk size to {new_chunk_size} due to memory constraints",
                )
            else:
                # Try to process in smaller batches
                return RecoveryResult(
                    success=True,
                    recovery_method="batch_processing",
                    warning_message="Switched to batch processing due to memory constraints",
                )

        except Exception as e:
            logger.error(f"Memory recovery failed: {e}")
            return RecoveryResult(success=False)

    def _recover_from_data_corruption(
        self, error_info: ErrorInfo, context: Dict[str, Any]
    ) -> RecoveryResult:
        """Recover from data corruption errors."""
        logger.info("Attempting data corruption recovery")

        try:
            # Try to identify and skip corrupted segments
            if "signal" in context:
                signal = context["signal"]

                # Check for NaN or Inf values
                if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
                    # Replace with interpolated values
                    valid_mask = ~(np.isnan(signal) | np.isinf(signal))

                    if np.any(valid_mask):
                        # Interpolate missing values
                        from scipy import interpolate

                        valid_indices = np.where(valid_mask)[0]
                        valid_values = signal[valid_mask]

                        if len(valid_indices) > 1:
                            f = interpolate.interp1d(
                                valid_indices,
                                valid_values,
                                kind="linear",
                                fill_value="extrapolate",
                            )
                            signal[~valid_mask] = f(np.where(~valid_mask)[0])

                            return RecoveryResult(
                                success=True,
                                recovered_data=signal,
                                recovery_method="interpolation",
                                warning_message="Interpolated corrupted data points",
                            )

            return RecoveryResult(success=False)

        except Exception as e:
            logger.error(f"Data corruption recovery failed: {e}")
            return RecoveryResult(success=False)

    def _recover_from_timeout(
        self, error_info: ErrorInfo, context: Dict[str, Any]
    ) -> RecoveryResult:
        """Recover from processing timeout errors."""
        logger.info("Attempting timeout recovery")

        try:
            # Increase timeout or reduce processing complexity
            if "timeout" in context:
                original_timeout = context["timeout"]
                new_timeout = original_timeout * 2
                context["timeout"] = new_timeout

                return RecoveryResult(
                    success=True,
                    recovery_method="timeout_extension",
                    warning_message=f"Increased timeout to {new_timeout}s",
                )
            else:
                # Try simpler processing approach
                return RecoveryResult(
                    success=True,
                    recovery_method="simplified_processing",
                    warning_message="Switched to simplified processing due to timeout",
                )

        except Exception as e:
            logger.error(f"Timeout recovery failed: {e}")
            return RecoveryResult(success=False)

    def _recover_from_file_error(
        self, error_info: ErrorInfo, context: Dict[str, Any]
    ) -> RecoveryResult:
        """Recover from file-related errors."""
        logger.info("Attempting file error recovery")

        try:
            # Try alternative file paths or formats
            if "file_path" in context:
                file_path = context["file_path"]

                # Try different file extensions
                base_path = Path(file_path).with_suffix("")
                extensions = [".csv", ".parquet", ".hdf5", ".npy"]

                for ext in extensions:
                    alt_path = base_path.with_suffix(ext)
                    if alt_path.exists():
                        context["file_path"] = str(alt_path)

                        return RecoveryResult(
                            success=True,
                            recovery_method="alternative_file_format",
                            warning_message=f"Using alternative file format: {ext}",
                        )

            return RecoveryResult(success=False)

        except Exception as e:
            logger.error(f"File error recovery failed: {e}")
            return RecoveryResult(success=False)

    def _recover_from_config_error(
        self, error_info: ErrorInfo, context: Dict[str, Any]
    ) -> RecoveryResult:
        """Recover from configuration errors."""
        logger.info("Attempting configuration error recovery")

        try:
            # Use default configuration values
            if "config" in context:
                config = context["config"]

                # Reset to default values
                config.reset_to_defaults()

                return RecoveryResult(
                    success=True,
                    recovery_method="default_configuration",
                    warning_message="Reset to default configuration values",
                )

            return RecoveryResult(success=False)

        except Exception as e:
            logger.error(f"Configuration error recovery failed: {e}")
            return RecoveryResult(success=False)

    def _recover_from_network_error(
        self, error_info: ErrorInfo, context: Dict[str, Any]
    ) -> RecoveryResult:
        """Recover from network-related errors."""
        logger.info("Attempting network error recovery")

        try:
            # Retry with exponential backoff
            if "retry_count" not in context:
                context["retry_count"] = 0

            context["retry_count"] += 1

            if context["retry_count"] <= 3:
                # Wait before retry
                wait_time = 2 ** context["retry_count"]
                time.sleep(wait_time)

                return RecoveryResult(
                    success=True,
                    recovery_method="retry_with_backoff",
                    warning_message=f"Retrying network operation (attempt {context['retry_count']})",
                )
            else:
                return RecoveryResult(success=False)

        except Exception as e:
            logger.error(f"Network error recovery failed: {e}")
            return RecoveryResult(success=False)

    def _recover_from_generic_error(
        self, error_info: ErrorInfo, context: Dict[str, Any]
    ) -> RecoveryResult:
        """Recover from generic errors."""
        logger.info("Attempting generic error recovery")

        try:
            # Try to preserve partial results
            if "partial_results" in context:
                partial_results = context["partial_results"]

                return RecoveryResult(
                    success=True,
                    recovery_method="partial_results_preservation",
                    partial_results=partial_results,
                    warning_message="Preserved partial results from processing",
                )

            return RecoveryResult(success=False)

        except Exception as e:
            logger.error(f"Generic error recovery failed: {e}")
            return RecoveryResult(success=False)

    def save_partial_results(self, session_id: str, results: Dict[str, Any]) -> None:
        """
        Save partial results for potential recovery.

        Args:
            session_id: Session identifier
            results: Partial results to save
        """
        self.partial_results[session_id] = {
            "timestamp": datetime.now(),
            "results": results,
        }

        logger.debug(f"Saved partial results for session: {session_id}")

    def get_partial_results(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get partial results for a session.

        Args:
            session_id: Session identifier

        Returns:
            Partial results or None if not found
        """
        return self.partial_results.get(session_id)

    def cleanup_partial_results(self, session_id: str) -> None:
        """
        Clean up partial results for a session.

        Args:
            session_id: Session identifier
        """
        if session_id in self.partial_results:
            del self.partial_results[session_id]

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error recovery statistics.

        Returns:
            Error statistics dictionary
        """
        total_errors = len(self.recovery_history)
        successful_recoveries = sum(1 for r in self.recovery_history if r.success)
        recovery_success_rate = successful_recoveries / max(total_errors, 1)

        return {
            "total_errors": total_errors,
            "recovery_attempts": total_errors,
            "successful_recoveries": successful_recoveries,
            "recovery_success_rate": recovery_success_rate,
            "partial_results_saved": len(self.partial_results),
            "available_strategies": list(self.recovery_strategies.keys()),
        }


class ErrorHandler:
    """
    Comprehensive error handling system with user-friendly error messages.
    """

    def __init__(
        self,
        config_manager: Optional[DynamicConfigManager] = None,
        log_file: Optional[str] = None,
    ):
        """
        Initialize error handler.

        Args:
            config_manager: Configuration manager instance
            log_file: Optional log file path
        """
        self.config = config_manager or DynamicConfigManager()
        self.recovery_manager = ErrorRecoveryManager(self.config)
        self.error_history = []
        self._lock = threading.Lock()

        # Setup logging
        if log_file:
            self._setup_error_logging(log_file)

        # Error message templates
        self.error_templates = self._load_error_templates()

    def _setup_error_logging(self, log_file: str) -> None:
        """Setup error-specific logging."""
        error_logger = logging.getLogger("vitaldsp_errors")
        error_logger.setLevel(logging.ERROR)

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.ERROR)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add handler to logger
        error_logger.addHandler(file_handler)

    def _load_error_templates(self) -> Dict[str, str]:
        """Load user-friendly error message templates."""
        return {
            "memory_error": "The system ran out of memory while processing your data. "
            "Try processing smaller chunks or reducing the data size.",
            "data_corruption": "The data appears to be corrupted or contains invalid values. "
            "Please check your data file and try again.",
            "processing_timeout": "The processing took longer than expected and timed out. "
            "Try reducing the data size or using simpler processing options.",
            "file_not_found": "The specified file could not be found. "
            "Please check the file path and ensure the file exists.",
            "configuration_error": "There was an error with the processing configuration. "
            "The system will use default settings.",
            "network_error": "A network error occurred. Please check your internet connection "
            "and try again.",
            "generic_error": "An unexpected error occurred during processing. "
            "Please try again or contact support if the problem persists.",
        }

    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.PROCESSING,
        recovery_attempted: bool = False,
    ) -> ErrorInfo:
        """
        Handle an error and create error information.

        Args:
            error: Exception that occurred
            context: Processing context
            severity: Error severity level
            category: Error category
            recovery_attempted: Whether recovery was attempted

        Returns:
            Error information
        """
        error_id = self._generate_error_id()

        # Create error information
        error_info = ErrorInfo(
            error_id=error_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            error_type=type(error).__name__,
            message=str(error),
            details=self._extract_error_details(error),
            context=context,
            stack_trace=traceback.format_exc(),
            recovery_attempted=recovery_attempted,
        )

        # Log error
        self._log_error(error_info)

        # Store in history
        with self._lock:
            self.error_history.append(error_info)

        return error_info

    def _generate_error_id(self) -> str:
        """Generate unique error ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = str(int(time.time() * 1000))[-6:]
        return f"ERR_{timestamp}_{random_suffix}"

    def _extract_error_details(self, error: Exception) -> Dict[str, Any]:
        """Extract detailed information from error."""
        details = {
            "error_type": type(error).__name__,
            "error_module": getattr(error, "__module__", "unknown"),
            "error_args": getattr(error, "args", []),
        }

        # Add specific details for common error types
        if isinstance(error, MemoryError):
            details["memory_info"] = self._get_memory_info()
        elif isinstance(error, FileNotFoundError):
            details["file_path"] = getattr(error, "filename", "unknown")
        elif isinstance(error, ValueError):
            details["value"] = getattr(error, "value", "unknown")

        return details

    def _get_memory_info(self) -> Dict[str, Any]:
        """Get current memory information."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            return {
                "total_memory_gb": memory.total / (1024**3),
                "available_memory_gb": memory.available / (1024**3),
                "used_memory_gb": memory.used / (1024**3),
                "memory_percent": memory.percent,
            }
        except ImportError:
            return {"error": "psutil not available"}

    def _log_error(self, error_info: ErrorInfo) -> None:
        """Log error information."""
        error_logger = logging.getLogger("vitaldsp_errors")

        log_message = (
            f"Error {error_info.error_id}: {error_info.error_type} - "
            f"{error_info.message} (Severity: {error_info.severity.value}, "
            f"Category: {error_info.category.value})"
        )

        if error_info.severity == ErrorSeverity.CRITICAL:
            error_logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            error_logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            error_logger.warning(log_message)
        else:
            error_logger.info(log_message)

    def get_user_friendly_message(self, error_info: ErrorInfo) -> str:
        """
        Get user-friendly error message.

        Args:
            error_info: Error information

        Returns:
            User-friendly error message
        """
        # Get template for error type
        template = self.error_templates.get(
            error_info.error_type.lower(), self.error_templates["generic_error"]
        )

        # Add recovery information if available
        if error_info.recovery_attempted:
            if error_info.recovery_successful:
                template += (
                    " The system has automatically recovered and continued processing."
                )
            else:
                template += " Automatic recovery was attempted but failed."

        return template

    def attempt_error_recovery(
        self, error_info: ErrorInfo, context: Dict[str, Any]
    ) -> RecoveryResult:
        """
        Attempt to recover from an error.

        Args:
            error_info: Error information
            context: Processing context

        Returns:
            Recovery result
        """
        logger.info(f"Attempting recovery for error: {error_info.error_id}")

        # Attempt recovery
        recovery_result = self.recovery_manager.attempt_recovery(error_info, context)

        # Update error info
        error_info.recovery_attempted = True
        error_info.recovery_successful = recovery_result.success
        error_info.recovery_method = recovery_result.recovery_method

        return recovery_result

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics.

        Returns:
            Error statistics
        """
        if not self.error_history:
            return {"total_errors": 0}

        # Calculate statistics
        total_errors = len(self.error_history)

        severity_counts = {}
        category_counts = {}
        error_type_counts = {}

        for error in self.error_history:
            severity_counts[error.severity.value] = (
                severity_counts.get(error.severity.value, 0) + 1
            )
            category_counts[error.category.value] = (
                category_counts.get(error.category.value, 0) + 1
            )
            error_type_counts[error.error_type] = (
                error_type_counts.get(error.error_type, 0) + 1
            )

        # Calculate recovery statistics
        recovery_attempts = sum(1 for e in self.error_history if e.recovery_attempted)
        successful_recoveries = sum(
            1 for e in self.error_history if e.recovery_successful
        )

        return {
            "total_errors": total_errors,
            "severity_distribution": severity_counts,
            "category_distribution": category_counts,
            "error_type_distribution": error_type_counts,
            "recovery_attempts": recovery_attempts,
            "successful_recoveries": successful_recoveries,
            "recovery_success_rate": successful_recoveries / max(recovery_attempts, 1),
        }

    def generate_error_report(self, session_id: Optional[str] = None) -> str:
        """
        Generate comprehensive error report.

        Args:
            session_id: Optional session ID to filter errors

        Returns:
            Error report
        """
        report = []
        report.append("=== Error Report ===")
        report.append("")

        # Filter errors by session if specified
        if session_id:
            session_errors = [
                e
                for e in self.error_history
                if session_id in e.context.get("session_id", "")
            ]
        else:
            session_errors = self.error_history

        if not session_errors:
            report.append("No errors recorded.")
            return "\n".join(report)

        # Error summary
        report.append(f"Total Errors: {len(session_errors)}")
        report.append("")

        # Error details
        for error in session_errors:
            report.append(f"Error ID: {error.error_id}")
            report.append(f"Timestamp: {error.timestamp}")
            report.append(f"Type: {error.error_type}")
            report.append(f"Severity: {error.severity.value}")
            report.append(f"Category: {error.category.value}")
            report.append(f"Message: {error.message}")

            if error.recovery_attempted:
                report.append(
                    f"Recovery: {'Successful' if error.recovery_successful else 'Failed'}"
                )
                if error.recovery_method:
                    report.append(f"Recovery Method: {error.recovery_method}")

            report.append("")

        return "\n".join(report)

    def clear_error_history(self) -> None:
        """Clear error history."""
        with self._lock:
            self.error_history.clear()
            logger.info("Error history cleared")


def error_handler(
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.PROCESSING,
    recovery_enabled: bool = True,
):
    """
    Decorator for automatic error handling.

    Args:
        severity: Default error severity
        category: Default error category
        recovery_enabled: Whether to enable automatic recovery

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler_instance = ErrorHandler()

            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create context
                context = {
                    "function": func.__name__,
                    "args": str(args)[:200],  # Limit length
                    "kwargs": str(kwargs)[:200],
                }

                # Handle error
                error_info = error_handler_instance.handle_error(
                    e, context, severity, category
                )

                # Attempt recovery if enabled
                if recovery_enabled:
                    recovery_result = error_handler_instance.attempt_error_recovery(
                        error_info, context
                    )

                    if recovery_result.success:
                        logger.info(f"Recovery successful for {func.__name__}")
                        return recovery_result.recovered_data

                # Re-raise if recovery failed or disabled
                raise

        return wrapper

    return decorator


class RobustProcessingPipeline:
    """
    Robust processing pipeline with comprehensive error handling and recovery.
    """

    def __init__(self, config_manager: Optional[DynamicConfigManager] = None):
        """
        Initialize robust processing pipeline.

        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager or DynamicConfigManager()
        self.error_handler = ErrorHandler(self.config)
        self.recovery_manager = ErrorRecoveryManager(self.config)

        # Processing statistics
        self.stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "recovered_operations": 0,
            "total_processing_time": 0.0,
        }

    @error_handler(severity=ErrorSeverity.HIGH, category=ErrorCategory.PROCESSING)
    def process_with_error_handling(
        self, processing_func: Callable, *args, **kwargs
    ) -> Any:
        """
        Process with comprehensive error handling.

        Args:
            processing_func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Processing result
        """
        start_time = time.time()
        self.stats["total_operations"] += 1

        try:
            result = processing_func(*args, **kwargs)
            self.stats["successful_operations"] += 1
            return result

        except Exception as e:
            self.stats["failed_operations"] += 1

            # Create error context
            context = {
                "function": processing_func.__name__,
                "args": str(args)[:200],
                "kwargs": str(kwargs)[:200],
                "processing_time": time.time() - start_time,
            }

            # Handle error
            error_info = self.error_handler.handle_error(
                e, context, ErrorSeverity.HIGH, ErrorCategory.PROCESSING
            )

            # Attempt recovery
            recovery_result = self.error_handler.attempt_error_recovery(
                error_info, context
            )

            if recovery_result.success:
                self.stats["recovered_operations"] += 1
                logger.info(f"Recovery successful for {processing_func.__name__}")
                return recovery_result.recovered_data
            else:
                # Re-raise if recovery failed
                raise

        finally:
            self.stats["total_processing_time"] += time.time() - start_time

    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Processing statistics
        """
        success_rate = self.stats["successful_operations"] / max(
            self.stats["total_operations"], 1
        )
        recovery_rate = self.stats["recovered_operations"] / max(
            self.stats["failed_operations"], 1
        )

        return {
            "processing_stats": self.stats,
            "success_rate": success_rate,
            "recovery_rate": recovery_rate,
            "error_stats": self.error_handler.get_error_statistics(),
        }

    def generate_processing_report(self) -> str:
        """
        Generate comprehensive processing report.

        Returns:
            Processing report
        """
        report = []
        report.append("=== Robust Processing Report ===")
        report.append("")

        # Processing statistics
        stats = self.get_processing_statistics()
        report.append(
            f"Total Operations: {stats['processing_stats']['total_operations']}"
        )
        report.append(
            f"Successful Operations: {stats['processing_stats']['successful_operations']}"
        )
        report.append(
            f"Failed Operations: {stats['processing_stats']['failed_operations']}"
        )
        report.append(
            f"Recovered Operations: {stats['processing_stats']['recovered_operations']}"
        )
        report.append(f"Success Rate: {stats['success_rate']:.2%}")
        report.append(f"Recovery Rate: {stats['recovery_rate']:.2%}")
        report.append(
            f"Total Processing Time: {stats['processing_stats']['total_processing_time']:.2f}s"
        )
        report.append("")

        # Error statistics
        error_stats = stats["error_stats"]
        report.append(f"Total Errors: {error_stats['total_errors']}")
        report.append(
            f"Recovery Success Rate: {error_stats['recovery_success_rate']:.2%}"
        )
        report.append("")

        return "\n".join(report)
