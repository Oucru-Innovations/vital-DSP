"""
Validation utilities for the PPG analysis tool.

This module provides validation functions for various inputs including
file paths, data parameters, and signal processing parameters.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .exceptions import (
    FileNotFoundError,
    FileTooLargeError,
    InvalidFileFormatError,
    InvalidParameterError,
)
from .exceptions import ValidationError as PPGValidationError


def validate_file_path(file_path: str) -> Path:
    """
    Validate and normalize a file path.

    Args:
        file_path: Path to the file

    Returns:
        Normalized Path object

    Raises:
        FileNotFoundError: If file doesn't exist
        InvalidParameterError: If path is invalid
    """
    if not file_path or not isinstance(file_path, str):
        raise InvalidParameterError("File path must be a non-empty string")

    try:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}", file_path=file_path)
        return path
    except Exception as e:
        raise InvalidParameterError(f"Invalid file path: {file_path}", value=file_path) from e


def validate_file_size(file_path: Union[str, Path], max_size_mb: int) -> None:
    """
    Validate that a file doesn't exceed the maximum size.

    Args:
        file_path: Path to the file
        max_size_mb: Maximum file size in MB

    Raises:
        FileTooLargeError: If file exceeds size limit
    """
    path = Path(file_path) if isinstance(file_path, str) else file_path
    file_size_mb = path.stat().st_size / (1024 * 1024)

    if file_size_mb > max_size_mb:
        raise FileTooLargeError(
            f"File size ({file_size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)", file_path=str(path)
        )


def validate_csv_file(file_path: Union[str, Path]) -> None:
    """
    Validate that a file is a valid CSV file.

    Args:
        file_path: Path to the file

    Raises:
        InvalidFileFormatError: If file is not a valid CSV
    """
    path = Path(file_path) if isinstance(file_path, str) else file_path

    # Check file extension
    if path.suffix.lower() != ".csv":
        raise InvalidFileFormatError(
            f"File must have .csv extension, got: {path.suffix}", file_path=str(path)
        )

    # Try to read the file as CSV
    try:
        pd.read_csv(path, nrows=1)
    except Exception as e:
        raise InvalidFileFormatError(f"File is not a valid CSV: {e}", file_path=str(path))


def validate_dataframe(
    df: pd.DataFrame, required_columns: Optional[List[str]] = None, min_rows: int = 1
) -> None:
    """
    Validate a pandas DataFrame.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required

    Raises:
        PPGValidationError: If DataFrame is invalid
    """
    if not isinstance(df, pd.DataFrame):
        raise PPGValidationError("Input must be a pandas DataFrame")

    if len(df) < min_rows:
        raise PPGValidationError(f"DataFrame must have at least {min_rows} rows, got {len(df)}")

    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise PPGValidationError(f"Missing required columns: {missing_cols}", field="columns")


def validate_numeric_array(
    arr: np.ndarray, min_length: int = 1, allow_nan: bool = False, allow_inf: bool = False
) -> None:
    """
    Validate a numeric numpy array.

    Args:
        arr: Array to validate
        min_length: Minimum array length
        allow_nan: Whether to allow NaN values
        allow_inf: Whether to allow infinite values

    Raises:
        PPGValidationError: If array is invalid
    """
    if not isinstance(arr, np.ndarray):
        raise PPGValidationError("Input must be a numpy array")

    if arr.size < min_length:
        raise PPGValidationError(f"Array must have at least {min_length} elements, got {arr.size}")

    if not np.issubdtype(arr.dtype, np.number):
        raise PPGValidationError("Array must contain numeric values")

    if not allow_nan and np.any(np.isnan(arr)):
        raise PPGValidationError("Array contains NaN values")

    if not allow_inf and np.any(np.isinf(arr)):
        raise PPGValidationError("Array contains infinite values")


def validate_sampling_frequency(fs: float) -> None:
    """
    Validate sampling frequency parameter.

    Args:
        fs: Sampling frequency in Hz

    Raises:
        InvalidParameterError: If sampling frequency is invalid
    """
    if not isinstance(fs, (int, float)) or fs <= 0:
        raise InvalidParameterError(
            "Sampling frequency must be a positive number", field="fs", value=fs
        )

    if fs > 10000:  # Reasonable upper limit for PPG
        raise InvalidParameterError(
            "Sampling frequency seems unreasonably high for PPG data", field="fs", value=fs
        )


def validate_heart_rate_range(hr_min: float, hr_max: float) -> None:
    """
    Validate heart rate range parameters.

    Args:
        hr_min: Minimum heart rate in BPM
        hr_max: Maximum heart rate in BPM

    Raises:
        InvalidParameterError: If heart rate range is invalid
    """
    if not isinstance(hr_min, (int, float)) or hr_min < 20:
        raise InvalidParameterError(
            "Minimum heart rate must be at least 20 BPM", field="hr_min", value=hr_min
        )

    if not isinstance(hr_max, (int, float)) or hr_max > 300:
        raise InvalidParameterError(
            "Maximum heart rate must be at most 300 BPM", field="hr_max", value=hr_max
        )

    if hr_min >= hr_max:
        raise InvalidParameterError(
            "Minimum heart rate must be less than maximum heart rate",
            field="heart_rate_range",
            value=(hr_min, hr_max),
        )


def validate_window_parameters(
    start: int, end: int, total_rows: int, min_window_size: int = 10
) -> None:
    """
    Validate window parameters for data slicing.

    Args:
        start: Start row index
        end: End row index
        total_rows: Total number of rows in dataset
        min_window_size: Minimum window size in rows

    Raises:
        InvalidParameterError: If window parameters are invalid
    """
    if not isinstance(start, int) or start < 0:
        raise InvalidParameterError(
            "Start index must be a non-negative integer", field="start", value=start
        )

    if not isinstance(end, int) or end < start:
        raise InvalidParameterError(
            "End index must be an integer greater than start", field="end", value=end
        )

    if end > total_rows:
        raise InvalidParameterError(
            f"End index ({end}) exceeds total rows ({total_rows})", field="end", value=end
        )

    window_size = end - start
    if window_size < min_window_size:
        raise InvalidParameterError(
            f"Window size ({window_size}) must be at least {min_window_size} rows",
            field="window_size",
            value=window_size,
        )


def validate_column_mapping(
    columns: List[str], red_col: Optional[str], ir_col: Optional[str]
) -> None:
    """
    Validate column mapping for PPG data.

    Args:
        columns: Available column names
        red_col: Selected RED channel column
        ir_col: Selected IR channel column

    Raises:
        PPGValidationError: If column mapping is invalid
    """
    if not isinstance(columns, list) or not columns:
        raise PPGValidationError("Columns list must be non-empty")

    if red_col and red_col not in columns:
        raise PPGValidationError(
            f"RED column '{red_col}' not found in available columns", field="red_col", value=red_col
        )

    if ir_col and ir_col not in columns:
        raise PPGValidationError(
            f"IR column '{ir_col}' not found in available columns", field="ir_col", value=ir_col
        )

    if red_col and ir_col and red_col == ir_col:
        raise PPGValidationError(
            "RED and IR columns must be different", field="column_mapping", value=(red_col, ir_col)
        )
