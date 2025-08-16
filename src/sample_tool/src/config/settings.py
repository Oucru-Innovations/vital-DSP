"""
Configuration settings and constants for the PPG analysis tool.

This module provides a centralized configuration system with environment variable
support, validation, and organized constants for the PPG analysis application.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    # Try Pydantic v2 first
    from pydantic import BaseModel, Field, field_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict

    PYDANTIC_V2 = True
except ImportError:
    try:
        # Fallback to Pydantic v1
        from pydantic import BaseSettings, Field, validator

        PYDANTIC_V2 = False
    except ImportError:
        # Fallback to simple dataclass if no Pydantic available
        PYDANTIC_V2 = False
        PYDANTIC_AVAILABLE = False


if PYDANTIC_V2:

    class AppSettings(BaseSettings):
        """Application configuration settings with environment variable support."""

        # App metadata
        app_name: str = Field(default="PPG Filter Lab", description="Application name")
        app_version: str = Field(default="1.0.0", description="Application version")
        debug: bool = Field(default=False, description="Debug mode")

        # File handling
        default_file_on_disk: str = Field(default="PPG.csv", description="Default file to look for")
        max_file_size_mb: int = Field(default=100, description="Maximum file size in MB")
        temp_file_prefix: str = Field(default="ppg_", description="Prefix for temporary files")

        # Signal processing defaults
        default_fs: float = Field(default=100.0, description="Default sampling frequency")
        default_window_start: int = Field(default=0, description="Default window start")
        default_window_end: int = Field(default=9_999, description="Default window end")
        max_display_points: int = Field(default=300_000, description="Maximum display points")
        default_decim_user: int = Field(default=1, description="Default decimation factor")

        # Heart rate and peak detection
        default_hr_min: int = Field(default=40, description="Minimum heart rate")
        default_hr_max: int = Field(default=180, description="Maximum heart rate")
        default_peak_prom_factor: float = Field(default=0.5, description="Peak prominence factor")

        # Spectrogram settings
        default_spec_win_sec: float = Field(default=2.0, description="Spectrogram window size")
        default_spec_overlap: float = Field(default=0.5, description="Spectrogram overlap")

        # UI settings
        grid_columns: str = Field(default="340px 2.1fr 0.9fr", description="Grid column layout")
        grid_gap: str = Field(default="18px", description="Grid gap")
        max_width: str = Field(default="1760px", description="Maximum width")

        # Validation
        @field_validator("default_fs")
        @classmethod
        def validate_fs(cls, v):
            if v <= 0:
                raise ValueError("Sampling frequency must be positive")
            return v

        @field_validator("default_hr_min", "default_hr_max")
        @classmethod
        def validate_hr_range(cls, v):
            if not (20 <= v <= 300):
                raise ValueError("Heart rate must be between 20 and 300 BPM")
            return v

        @field_validator("default_peak_prom_factor")
        @classmethod
        def validate_peak_factor(cls, v):
            if not (0.1 <= v <= 2.0):
                raise ValueError("Peak prominence factor must be between 0.1 and 2.0")
            return v

        @field_validator("default_spec_overlap")
        @classmethod
        def validate_overlap(cls, v):
            if not (0.0 <= v < 1.0):
                raise ValueError("Spectrogram overlap must be between 0.0 and 1.0")
            return v

        model_config = SettingsConfigDict(
            env_file=".env", env_file_encoding="utf-8", case_sensitive=False
        )

elif not PYDANTIC_V2 and "PYDANTIC_AVAILABLE" not in locals():

    class AppSettings(BaseSettings):
        """Application configuration settings with environment variable support."""

        # App metadata
        app_name: str = Field(default="PPG Filter Lab", env="APP_NAME")
        app_version: str = Field(default="1.0.0", env="APP_VERSION")
        debug: bool = Field(default=False, env="DEBUG")

        # File handling
        default_file_on_disk: str = Field(default="PPG.csv", env="DEFAULT_FILE_ON_DISK")
        max_file_size_mb: int = Field(default=100, env="MAX_FILE_SIZE_MB")
        temp_file_prefix: str = Field(default="ppg_", env="TEMP_FILE_PREFIX")

        # Signal processing defaults
        default_fs: float = Field(default=100.0, env="DEFAULT_FS")
        default_window_start: int = Field(default=0, env="DEFAULT_WINDOW_START")
        default_window_end: int = Field(default=9_999, env="DEFAULT_WINDOW_END")
        max_display_points: int = Field(default=300_000, env="MAX_DISPLAY_POINTS")
        default_decim_user: int = Field(default=1, env="DEFAULT_DECIM_USER")

        # Heart rate and peak detection
        default_hr_min: int = Field(default=40, env="DEFAULT_HR_MIN")
        default_hr_max: int = Field(default=180, env="DEFAULT_HR_MAX")
        default_peak_prom_factor: float = Field(default=0.5, env="DEFAULT_PEAK_PROM_FACTOR")

        # Spectrogram settings
        default_spec_win_sec: float = Field(default=2.0, env="DEFAULT_SPEC_WIN_SEC")
        default_spec_overlap: float = Field(default=0.5, env="DEFAULT_SPEC_OVERLAP")

        # UI settings
        grid_columns: str = Field(default="340px 2.1fr 0.9fr", env="GRID_COLUMNS")
        grid_gap: str = Field(default="18px", env="GRID_GAP")
        max_width: str = Field(default="1760px", env="MAX_WIDTH")

        # Validation
        @validator("default_fs")
        def validate_fs(cls, v):
            if v <= 0:
                raise ValueError("Sampling frequency must be positive")
            return v

        @validator("default_hr_min", "default_hr_max")
        def validate_hr_range(cls, v):
            if not (20 <= v <= 300):
                raise ValueError("Heart rate must be between 20 and 300 BPM")
            return v

        @validator("default_peak_prom_factor")
        def validate_peak_factor(cls, v):
            if not (0.1 <= v <= 2.0):
                raise ValueError("Peak prominence factor must be between 0.1 and 2.0")
            return v

        @validator("default_spec_overlap")
        def validate_overlap(cls, v):
            if not (0.0 <= v < 1.0):
                raise ValueError("Spectrogram overlap must be between 0.0 and 1.0")
            return v

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False

else:
    # Fallback to simple dataclass if no Pydantic available
    @dataclass
    class AppSettings:
        """Application configuration settings (fallback without Pydantic)."""

        # App metadata
        app_name: str = "PPG Filter Lab"
        app_version: str = "1.0.0"
        debug: bool = False

        # File handling
        default_file_on_disk: str = "PPG.csv"
        max_file_size_mb: int = 100
        temp_file_prefix: str = "ppg_"

        # Signal processing defaults
        default_fs: float = 100.0
        default_window_start: int = 0
        default_window_end: int = 9_999
        max_display_points: int = 300_000
        default_decim_user: int = 1

        # Heart rate and peak detection
        default_hr_min: int = 40
        default_hr_max: int = 180
        default_peak_prom_factor: float = 0.5

        # Spectrogram settings
        default_spec_win_sec: float = 2.0
        default_spec_overlap: float = 0.5

        # UI settings
        grid_columns: str = "340px 2.1fr 0.9fr"
        grid_gap: str = "18px"
        max_width: str = "1760px"


# Create settings instance
try:
    settings = AppSettings()
except Exception as e:
    # Fallback to default values if settings creation fails
    print(f"Warning: Failed to load settings from environment: {e}")
    print("Using default configuration values.")
    settings = AppSettings()

# Legacy constants for backward compatibility
DEFAULT_FS = settings.default_fs
DEFAULT_FILE_ON_DISK = settings.default_file_on_disk
DEFAULT_WINDOW_START = settings.default_window_start
DEFAULT_WINDOW_END = settings.default_window_end
MAX_DISPLAY_POINTS = settings.max_display_points
DEFAULT_DECIM_USER = settings.default_decim_user
DEFAULT_HR_MIN = settings.default_hr_min
DEFAULT_HR_MAX = settings.default_hr_max
DEFAULT_PEAK_PROM_FACTOR = settings.default_peak_prom_factor
DEFAULT_SPEC_WIN_SEC = settings.default_spec_win_sec
DEFAULT_SPEC_OVERLAP = settings.default_spec_overlap
GRID_COLUMNS = settings.grid_columns
GRID_GAP = settings.grid_gap
MAX_WIDTH = settings.max_width

# App titles
APP_TITLE = f"{settings.app_name} — Window Mode (Wide + Insights)"
APP_SUBTITLE = "Load CSV • Select row window • Bigger plots • Extra insight visuals"

# File paths
SAMPLE_DATA_DIR = Path(__file__).parent.parent.parent / "sample_data"
DEFAULT_SAMPLE_DATA_PATH = SAMPLE_DATA_DIR / "toy_PPG_data.csv"

# Error messages
ERROR_MESSAGES = {
    "file_not_found": "File not found: {path}",
    "invalid_file_format": "Invalid file format. Please upload a CSV file.",
    "file_too_large": "File too large. Maximum size is {max_size}MB.",
    "invalid_column_mapping": "Invalid column mapping. Please select valid columns.",
    "data_loading_failed": "Failed to load data: {error}",
    "processing_error": "Error processing data: {error}",
}

# Success messages
SUCCESS_MESSAGES = {
    "file_loaded": "File loaded successfully: {filename}",
    "data_processed": "Data processed successfully",
    "export_completed": "Export completed successfully",
}

# UI constants
UI_CONSTANTS = {
    "upload_max_size": settings.max_file_size_mb * 1024 * 1024,  # Convert to bytes
    "temp_file_prefix": settings.temp_file_prefix,
    "default_placeholder": "e.g., PPG.csv",
    "button_styles": {
        "primary": "btn btn-primary",
        "secondary": "btn btn-secondary",
        "danger": "btn btn-danger",
    },
}
