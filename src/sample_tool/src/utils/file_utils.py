"""
File handling utilities for the PPG analysis tool.

This module provides functions for efficient file operations including:
- Quick row counting without loading entire files
- Column extraction for CSV files
- Handling of uploaded CSV content
- Windowed data reading for large datasets
- Automatic file detection
"""

import base64
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from ..config.settings import settings
from .exceptions import (
    FileNotFoundError,
    FileTooLargeError,
    InvalidFileFormatError,
    PPGError,
)
from .validation import (
    validate_csv_file,
    validate_file_path,
    validate_file_size,
)


def count_rows_quick(path: str) -> int:
    """
    Quickly count rows in a CSV file without loading data into memory.

    This function reads the file line by line to count rows, which is
    much more memory-efficient than loading the entire file for large datasets.

    Args:
        path: Path to the CSV file

    Returns:
        Number of data rows (excluding header)

    Raises:
        FileNotFoundError: If file doesn't exist
        InvalidFileFormatError: If file is not a valid CSV
        PPGError: For other file reading errors
    """
    try:
        # Validate file path and format
        file_path = validate_file_path(path)
        validate_csv_file(file_path)

        with open(file_path, "rb") as f:
            total_lines = sum(1 for _ in f)

        return max(0, total_lines - 1)  # Account for header row

    except (FileNotFoundError, InvalidFileFormatError):
        raise
    except Exception as e:
        raise PPGError(f"Failed to count rows in file: {e}", details={"path": path}) from e


def get_columns_only(path: str) -> List[str]:
    """
    Get column names from a CSV file without loading data.

    This function reads only the header row to extract column names,
    making it efficient for large files where only column information is needed.

    Args:
        path: Path to the CSV file

    Returns:
        List of column names as strings

    Raises:
        FileNotFoundError: If file doesn't exist
        InvalidFileFormatError: If file is not a valid CSV
        PPGError: For other file reading errors
    """
    try:
        # Validate file path and format
        file_path = validate_file_path(path)
        validate_csv_file(file_path)

        # Read only header row
        df_header = pd.read_csv(file_path, nrows=0)
        return list(df_header.columns)

    except (FileNotFoundError, InvalidFileFormatError):
        raise
    except Exception as e:
        raise PPGError(f"Failed to read columns from file: {e}", details={"path": path}) from e


def parse_uploaded_csv_to_temp(contents: str, filename: str) -> str:
    """
    Parse uploaded CSV content and save to temporary file.

    This function handles base64-encoded CSV content from web uploads
    and creates a temporary file for processing. The temporary file
    is automatically cleaned up by the system.

    Args:
        contents: Base64 encoded CSV content with data URL prefix
        filename: Original filename for determining file extension

    Returns:
        Path to the temporary file

    Raises:
        InvalidParameterError: If contents is empty or invalid
        PPGError: For other processing errors
    """
    if not contents:
        raise ValueError("Upload contents cannot be empty")

    try:
        # Parse base64 content and decode
        if "," not in contents:
            raise ValueError("Invalid upload content format")

        content_type, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)

        # Determine file extension
        suffix = ".csv" if not filename else f".{filename.split('.')[-1]}"
        if suffix.lower() != ".csv":
            suffix = ".csv"  # Force CSV extension for safety

        # Create temporary file with configured prefix
        fd, tmp_path = tempfile.mkstemp(prefix=settings.temp_file_prefix, suffix=suffix)

        with open(tmp_path, "wb") as f:
            f.write(decoded)

        return tmp_path

    except Exception as e:
        raise PPGError(f"Failed to parse uploaded CSV: {e}") from e


def read_window(path: str, cols: List[str], start_row: int, end_row: int) -> pd.DataFrame:
    """
    Read a specific window of rows from a CSV file.

    This function efficiently reads only a subset of rows from a large CSV file,
    making it suitable for handling datasets that don't fit in memory.

    Args:
        path: Path to the CSV file
        cols: List of column names to read
        start_row: Starting row index (0-based)
        end_row: Ending row index (exclusive)

    Returns:
        DataFrame containing the specified window of data

    Raises:
        FileNotFoundError: If file doesn't exist
        InvalidFileFormatError: If file is not a valid CSV
        PPGError: For other file reading errors
    """
    try:
        # Validate file path and format
        file_path = validate_file_path(path)
        validate_csv_file(file_path)

        # Validate window parameters
        if start_row < 0 or end_row <= start_row:
            raise ValueError("Invalid window parameters: start_row < 0 or end_row <= start_row")

        # Read the specified window
        df = pd.read_csv(
            file_path,
            usecols=cols,
            skiprows=range(1, start_row + 1),  # Skip header + rows before start
            nrows=end_row - start_row,
        )

        return df

    except (FileNotFoundError, InvalidFileFormatError):
        raise
    except Exception as e:
        raise PPGError(
            f"Failed to read data window: {e}",
            details={"path": path, "start_row": start_row, "end_row": end_row},
        ) from e


def get_auto_file_path(filename: str) -> Optional[str]:
    """
    Automatically detect and return the path to a file in the current directory.

    Args:
        filename: Name of the file to look for

    Returns:
        Full path to the file if found, None otherwise
    """
    try:
        # Look in current directory
        current_dir = Path.cwd()
        file_path = current_dir / filename

        if file_path.exists():
            return str(file_path.resolve())

        # Look in parent directories (up to 3 levels)
        for parent in current_dir.parents[:3]:
            file_path = parent / filename
            if file_path.exists():
                return str(file_path.resolve())

        return None

    except Exception:
        return None


def get_default_sample_data_path() -> Optional[str]:
    """
    Get the path to the default sample data file.

    Returns:
        Path to the sample data file if it exists, None otherwise
    """
    try:
        from ..config.settings import DEFAULT_SAMPLE_DATA_PATH

        if DEFAULT_SAMPLE_DATA_PATH.exists():
            return str(DEFAULT_SAMPLE_DATA_PATH.resolve())

        return None

    except Exception:
        return None


def validate_upload_file_size(contents: str, max_size_mb: int) -> None:
    """
    Validate that uploaded file content doesn't exceed size limits.

    Args:
        contents: Base64 encoded file content
        max_size_mb: Maximum file size in MB

    Raises:
        FileTooLargeError: If file exceeds size limit
    """
    try:
        if "," in contents:
            content_string = contents.split(",", 1)[1]
        else:
            content_string = contents

        # Estimate file size from base64 content
        decoded_size = len(base64.b64decode(content_string))
        file_size_mb = decoded_size / (1024 * 1024)

        if file_size_mb > max_size_mb:
            raise FileTooLargeError(
                f"Uploaded file size ({file_size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)"
            )

    except Exception as e:
        if isinstance(e, FileTooLargeError):
            raise
        raise PPGError(f"Failed to validate file size: {e}") from e


def cleanup_temp_file(file_path: str) -> None:
    """
    Clean up a temporary file.

    Args:
        file_path: Path to the temporary file to remove

    Note:
        This function safely removes temporary files and handles errors gracefully.
    """
    try:
        temp_path = Path(file_path)
        if temp_path.exists() and temp_path.is_file():
            temp_path.unlink()
    except Exception:
        # Ignore cleanup errors to avoid masking other issues
        pass
