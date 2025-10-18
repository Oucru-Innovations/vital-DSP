"""
Settings service for vitalDSP webapp.

This module provides comprehensive settings management including persistence,
validation, and integration with the application configuration.
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import pickle

logger = logging.getLogger(__name__)


@dataclass
class GeneralSettings:
    """General application settings."""

    theme: str = "light"
    timezone: str = "UTC"
    page_size: int = 25
    auto_refresh_interval: int = 30
    display_options: List[str] = None

    def __post_init__(self):
        if self.display_options is None:
            self.display_options = ["tooltips", "loading"]


@dataclass
class AnalysisSettings:
    """Analysis and signal processing settings."""

    default_sampling_freq: int = 1000
    default_fft_points: int = 1024
    default_window_type: str = "hann"
    peak_threshold: float = 0.5
    quality_threshold: float = 0.7
    analysis_options: List[str] = None

    def __post_init__(self):
        if self.analysis_options is None:
            self.analysis_options = ["auto_detect", "advanced_features"]


@dataclass
class DataSettings:
    """Data management and export settings."""

    max_file_size: int = 100
    auto_save_interval: int = 5
    data_retention_days: int = 30
    export_format: str = "csv"
    image_format: str = "png"
    export_options: List[str] = None

    def __post_init__(self):
        if self.export_options is None:
            self.export_options = ["metadata", "high_quality"]


@dataclass
class SystemSettings:
    """System performance and security settings."""

    max_cpu_usage: int = 80
    memory_limit_gb: int = 4
    parallel_threads: int = 4
    session_timeout_minutes: int = 60
    security_options: List[str] = None

    def __post_init__(self):
        if self.security_options is None:
            self.security_options = ["https", "encryption"]


@dataclass
class ApplicationSettings:
    """Complete application settings container."""

    general: GeneralSettings = None
    analysis: AnalysisSettings = None
    data: DataSettings = None
    system: SystemSettings = None
    last_updated: str = None
    version: str = "1.0.0"

    def __post_init__(self):
        if self.general is None:
            self.general = GeneralSettings()
        if self.analysis is None:
            self.analysis = AnalysisSettings()
        if self.data is None:
            self.data = DataSettings()
        if self.system is None:
            self.system = SystemSettings()
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()


class SettingsService:
    """Service for managing application settings."""

    def __init__(self, settings_file: str = "settings.json"):
        """Initialize the settings service."""
        self.settings_file = settings_file
        self.settings = self._load_settings()
        self._backup_settings()
        logger.info("Settings service initialized")

    def _load_settings(self) -> ApplicationSettings:
        """Load settings from file or create defaults."""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info(f"Settings loaded from {self.settings_file}")
                    return self._dict_to_settings(data)
            else:
                logger.info("No settings file found, creating default settings")
                return ApplicationSettings()
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            logger.info("Creating default settings due to error")
            return ApplicationSettings()

    def _save_settings(self) -> bool:
        """Save current settings to file."""
        try:
            # Create backup before saving
            self._backup_settings()

            # Save current settings
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(self._settings_to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(f"Settings saved to {self.settings_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return False

    def _backup_settings(self) -> None:
        """Create a backup of current settings."""
        try:
            backup_file = f"{self.settings_file}.backup"
            if os.path.exists(self.settings_file):
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    backup_data = f.read()
                with open(backup_file, "w", encoding="utf-8") as f:
                    f.write(backup_data)
                logger.debug("Settings backup created")
        except Exception as e:
            logger.warning(f"Could not create settings backup: {e}")

    def _settings_to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for JSON serialization."""
        settings_dict = asdict(self.settings)
        settings_dict["last_updated"] = datetime.now().isoformat()
        return settings_dict

    def _dict_to_settings(self, data: Dict[str, Any]) -> ApplicationSettings:
        """Convert dictionary to settings object."""
        try:
            # Handle general settings
            general_data = data.get("general", {})
            general = GeneralSettings(**general_data)

            # Handle analysis settings
            analysis_data = data.get("analysis", {})
            analysis = AnalysisSettings(**analysis_data)

            # Handle data settings
            data_data = data.get("data", {})
            data_settings = DataSettings(**data_data)

            # Handle system settings
            system_data = data.get("system", {})
            system = SystemSettings(**system_data)

            return ApplicationSettings(
                general=general,
                analysis=analysis,
                data=data_settings,
                system=system,
                last_updated=data.get("last_updated"),
                version=data.get("version", "1.0.0"),
            )
        except Exception as e:
            logger.error(f"Error converting dict to settings: {e}")
            return ApplicationSettings()

    def get_all_settings(self) -> ApplicationSettings:
        """Get all current settings."""
        return self.settings

    def get_general_settings(self) -> GeneralSettings:
        """Get general settings."""
        return self.settings.general

    def get_analysis_settings(self) -> AnalysisSettings:
        """Get analysis settings."""
        return self.settings.analysis

    def get_data_settings(self) -> DataSettings:
        """Get data settings."""
        return self.settings.data

    def get_system_settings(self) -> SystemSettings:
        """Get system settings."""
        return self.settings.system

    def update_general_settings(self, **kwargs) -> bool:
        """Update general settings."""
        try:
            for key, value in kwargs.items():
                if hasattr(self.settings.general, key):
                    setattr(self.settings.general, key, value)

            self.settings.last_updated = datetime.now().isoformat()
            return self._save_settings()
        except Exception as e:
            logger.error(f"Error updating general settings: {e}")
            return False

    def update_analysis_settings(self, **kwargs) -> bool:
        """Update analysis settings."""
        try:
            for key, value in kwargs.items():
                if hasattr(self.settings.analysis, key):
                    setattr(self.settings.analysis, key, value)

            self.settings.last_updated = datetime.now().isoformat()
            return self._save_settings()
        except Exception as e:
            logger.error(f"Error updating analysis settings: {e}")
            return False

    def update_data_settings(self, **kwargs) -> bool:
        """Update data settings."""
        try:
            for key, value in kwargs.items():
                if hasattr(self.settings.data, key):
                    setattr(self.settings.data, key, value)

            self.settings.last_updated = datetime.now().isoformat()
            return self._save_settings()
        except Exception as e:
            logger.error(f"Error updating data settings: {e}")
            return False

    def update_system_settings(self, **kwargs) -> bool:
        """Update system settings."""
        try:
            for key, value in kwargs.items():
                if hasattr(self.settings.system, key):
                    setattr(self.settings.system, key, value)

            self.settings.last_updated = datetime.now().isoformat()
            return self._save_settings()
        except Exception as e:
            logger.error(f"Error updating system settings: {e}")
            return False

    def reset_to_defaults(self) -> bool:
        """Reset all settings to default values."""
        try:
            self.settings = ApplicationSettings()
            return self._save_settings()
        except Exception as e:
            logger.error(f"Error resetting settings: {e}")
            return False

    def export_settings(self, format_type: str = "json") -> Optional[str]:
        """Export settings to specified format."""
        try:
            if format_type == "json":
                return json.dumps(
                    self._settings_to_dict(), indent=2, ensure_ascii=False
                )
            elif format_type == "pickle":
                return pickle.dumps(self.settings)
            else:
                logger.error(f"Unsupported export format: {format_type}")
                return None
        except Exception as e:
            logger.error(f"Error exporting settings: {e}")
            return None

    def import_settings(self, settings_data: str, format_type: str = "json") -> bool:
        """Import settings from specified format."""
        try:
            if format_type == "json":
                data = json.loads(settings_data)
                self.settings = self._dict_to_settings(data)
            elif format_type == "pickle":
                self.settings = pickle.loads(settings_data)
            else:
                logger.error(f"Unsupported import format: {format_type}")
                return False

            return self._save_settings()
        except Exception as e:
            logger.error(f"Error importing settings: {e}")
            return False

    def validate_settings(self) -> Dict[str, List[str]]:
        """Validate current settings and return any errors."""
        errors = {}

        # Validate general settings
        general_errors = []
        if self.settings.general.page_size not in [10, 25, 50, 100]:
            general_errors.append("Page size must be 10, 25, 50, or 100")
        if self.settings.general.auto_refresh_interval < 0:
            general_errors.append("Auto-refresh interval must be non-negative")
        if general_errors:
            errors["general"] = general_errors

        # Validate analysis settings
        analysis_errors = []
        if self.settings.analysis.default_sampling_freq < 100:
            analysis_errors.append("Sampling frequency must be at least 100 Hz")
        if self.settings.analysis.default_fft_points < 256:
            analysis_errors.append("FFT points must be at least 256")
        if (
            self.settings.analysis.peak_threshold < 0.1
            or self.settings.analysis.peak_threshold > 1.0
        ):
            analysis_errors.append("Peak threshold must be between 0.1 and 1.0")
        if analysis_errors:
            errors["analysis"] = analysis_errors

        # Validate data settings
        data_errors = []
        if (
            self.settings.data.max_file_size < 10
            or self.settings.data.max_file_size > 1000
        ):
            data_errors.append("Max file size must be between 10 and 1000 MB")
        if (
            self.settings.data.auto_save_interval < 1
            or self.settings.data.auto_save_interval > 60
        ):
            data_errors.append("Auto-save interval must be between 1 and 60 minutes")
        if data_errors:
            errors["data"] = data_errors

        # Validate system settings
        system_errors = []
        if (
            self.settings.system.max_cpu_usage < 10
            or self.settings.system.max_cpu_usage > 100
        ):
            system_errors.append("Max CPU usage must be between 10% and 100%")
        if (
            self.settings.system.memory_limit_gb < 1
            or self.settings.system.memory_limit_gb > 32
        ):
            system_errors.append("Memory limit must be between 1 and 32 GB")
        if system_errors:
            errors["system"] = system_errors

        return errors

    def get_settings_summary(self) -> Dict[str, Any]:
        """Get a summary of current settings for display."""
        return {
            "general": {
                "theme": self.settings.general.theme,
                "timezone": self.settings.general.timezone,
                "page_size": self.settings.general.page_size,
                "auto_refresh": self.settings.general.auto_refresh_interval,
            },
            "analysis": {
                "sampling_freq": self.settings.analysis.default_sampling_freq,
                "fft_points": self.settings.analysis.default_fft_points,
                "window_type": self.settings.analysis.default_window_type,
                "peak_threshold": self.settings.analysis.peak_threshold,
            },
            "data": {
                "max_file_size": self.settings.data.max_file_size,
                "auto_save": self.settings.data.auto_save_interval,
                "retention": self.settings.data.data_retention_days,
                "export_format": self.settings.data.export_format,
            },
            "system": {
                "cpu_usage": self.settings.system.max_cpu_usage,
                "memory_limit": self.settings.system.memory_limit_gb,
                "parallel_threads": self.settings.system.parallel_threads,
                "session_timeout": self.settings.system.session_timeout_minutes,
            },
            "last_updated": self.settings.last_updated,
            "version": self.settings.version,
        }


# Global settings service instance
_settings_service = None


def get_settings_service() -> SettingsService:
    """Get the global settings service instance."""
    global _settings_service
    if _settings_service is None:
        _settings_service = SettingsService()
    return _settings_service


def get_current_settings() -> ApplicationSettings:
    """Get current application settings."""
    return get_settings_service().get_all_settings()


def update_settings(category: str, **kwargs) -> bool:
    """Update settings for a specific category."""
    service = get_settings_service()

    if category == "general":
        return service.update_general_settings(**kwargs)
    elif category == "analysis":
        return service.update_analysis_settings(**kwargs)
    elif category == "data":
        return service.update_data_settings(**kwargs)
    elif category == "system":
        return service.update_system_settings(**kwargs)
    else:
        logger.error(f"Unknown settings category: {category}")
        return False


# Compatibility functions for tests
def load_settings() -> Dict[str, Any]:
    """Load settings from file (compatibility function)."""
    service = get_settings_service()
    settings = service.get_all_settings()
    # Convert ApplicationSettings to dict for compatibility
    return asdict(settings)


def save_settings(settings: Dict[str, Any]) -> bool:
    """Save settings to file (compatibility function)."""
    service = get_settings_service()
    # Convert dict to ApplicationSettings if needed
    if isinstance(settings, dict):
        # Create ApplicationSettings from dict
        app_settings = ApplicationSettings(**settings)
        return service.save_settings(app_settings)
    return service.save_settings(settings)


def get_default_settings() -> Dict[str, Any]:
    """Get default settings (compatibility function)."""
    # Create default ApplicationSettings
    default_settings = ApplicationSettings()
    return asdict(default_settings)


def validate_settings(settings: Dict[str, Any]) -> bool:
    """Validate settings (compatibility function)."""
    service = get_settings_service()
    try:
        # Convert dict to ApplicationSettings for validation
        if isinstance(settings, dict):
            app_settings = ApplicationSettings(**settings)
            # Call the service validation method and check if there are any errors
            errors = service.validate_settings()
            return len(errors) == 0  # Return True if no errors
        # Call the service validation method and check if there are any errors
        errors = service.validate_settings()
        return len(errors) == 0  # Return True if no errors
    except Exception:
        return False


def merge_settings(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge settings (compatibility function)."""
    # Simple dict merge for compatibility
    merged = {**base, **override}
    return merged