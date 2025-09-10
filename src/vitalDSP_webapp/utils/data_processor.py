"""
Data processor utility for vitalDSP webapp.

This module provides utility functions for data processing and validation.
"""

import pandas as pd
import numpy as np
import base64
import io
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class DataProcessor:
    """Utility class for data processing operations."""

    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        """Validate if file extension is supported."""
        if not filename:
            return False

        supported_extensions = [".csv", ".txt", ".mat"]
        file_ext = Path(filename).suffix.lower()
        return file_ext in supported_extensions

    @staticmethod
    def read_uploaded_content(contents: str, filename: str) -> Optional[pd.DataFrame]:
        """Read uploaded file content."""
        try:
            content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)

            if filename.endswith(".csv"):
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            elif filename.endswith(".txt"):
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep="\t")
            else:
                logger.error(f"Unsupported file format: {filename}")
                return None

            return df

        except Exception as e:
            logger.error(f"Error reading uploaded content: {e}")
            return None

    @staticmethod
    def read_file(file_path: str, filename: str) -> Optional[pd.DataFrame]:
        """Read file from file path."""
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif filename.endswith(".txt"):
                df = pd.read_csv(file_path, sep="\t")
            else:
                logger.error(f"Unsupported file format: {filename}")
                return None

            return df

        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return None

    @staticmethod
    def generate_sample_ppg_data(
        sampling_freq: float,
        duration: float = 10.0,
        heart_rate: float = 70,
        noise_level: float = 0.05,
    ) -> pd.DataFrame:
        """Generate sample PPG data for testing."""
        try:
            # Generate time axis
            t = np.arange(0, duration, 1 / sampling_freq)

            # Add input validation
            if sampling_freq <= 0:
                logger.error("Sampling frequency must be positive")
                return None
            if duration <= 0:
                logger.error("Duration must be positive")
                return None

            # Generate synthetic PPG signal
            heart_freq = heart_rate / 60  # Hz

            # Create PPG-like signal with multiple components
            signal = (
                1.0 * np.sin(2 * np.pi * heart_freq * t)  # Fundamental
                + 0.3 * np.sin(2 * np.pi * 2 * heart_freq * t)  # Second harmonic
                + 0.1 * np.sin(2 * np.pi * 3 * heart_freq * t)  # Third harmonic
                + noise_level * np.random.randn(len(t))  # Noise with configurable level
            )

            # Add respiratory modulation (0.2-0.5 Hz)
            resp_freq = 0.3  # Hz
            resp_modulation = 0.1 * np.sin(2 * np.pi * resp_freq * t)
            signal = signal * (1 + resp_modulation)

            # Create DataFrame
            df = pd.DataFrame(
                {
                    "time": t,  # Use lowercase 'time' to match test expectations
                    "signal": signal,  # Use 'signal' instead of 'PPG_Signal'
                }
            )

            return df

        except Exception as e:
            logger.error(f"Error generating sample data: {e}")
            return None

    @staticmethod
    def process_uploaded_data(
        df: pd.DataFrame,
        filename: str,
        sampling_freq: float,
        time_unit: str = "seconds",
    ) -> Optional[Dict[str, Any]]:
        """Process uploaded data and return metadata."""
        try:
            if df is None or df.empty:
                return None

            # Basic data validation
            if len(df.columns) < 2:
                logger.warning("Data should have at least 2 columns (time and signal)")

            # Calculate basic statistics
            signal_data = (
                df.iloc[:, 1].values if len(df.columns) > 1 else df.iloc[:, 0].values
            )

            # Convert time unit if needed
            if time_unit == "milliseconds":
                sampling_freq = sampling_freq / 1000
            elif time_unit == "minutes":
                sampling_freq = sampling_freq * 60

            duration = len(signal_data) / sampling_freq

            return {
                "filename": filename,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "sampling_freq": sampling_freq,
                "time_unit": time_unit,
                "duration": duration,
                "signal_length": len(signal_data),
                "mean": float(np.mean(signal_data)),
                "std": float(np.std(signal_data)),
                "min": float(np.min(signal_data)),
                "max": float(np.max(signal_data)),
                # Add fields expected by tests
                "num_rows": df.shape[0],
                "num_columns": df.shape[1],
            }

        except Exception as e:
            logger.error(f"Error processing uploaded data: {e}")
            return None
