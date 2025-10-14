"""
Settings utilities for vitalDSP webapp.

This module provides utility functions for settings management including
theme switching, system monitoring, and settings validation.
"""

import os
import psutil
import platform
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ThemeManager:
    """Manages application themes and styling."""

    THEMES = {
        "light": {
            "primary_color": "#3498db",
            "secondary_color": "#95a5a6",
            "success_color": "#27ae60",
            "warning_color": "#f39c12",
            "danger_color": "#e74c3c",
            "info_color": "#17a2b8",
            "background_color": "#ffffff",
            "text_color": "#2c3e50",
            "border_color": "#ecf0f1",
            "card_background": "#ffffff",
            "sidebar_background": "#f8f9fa",
        },
        "dark": {
            "primary_color": "#3498db",
            "secondary_color": "#95a5a6",
            "success_color": "#27ae60",
            "warning_color": "#f39c12",
            "danger_color": "#e74c3c",
            "info_color": "#17a2b8",
            "background_color": "#1a1a1a",
            "text_color": "#ffffff",
            "border_color": "#404040",
            "card_background": "#2d2d2d",
            "sidebar_background": "#1a1a1a",
        },
        "auto": {
            "primary_color": "#3498db",
            "secondary_color": "#95a5a6",
            "success_color": "#27ae60",
            "warning_color": "#f39c12",
            "danger_color": "#e74c3c",
            "info_color": "#17a2b8",
            "background_color": "var(--system-background)",
            "text_color": "var(--system-text)",
            "border_color": "var(--system-border)",
            "card_background": "var(--system-card)",
            "sidebar_background": "var(--system-sidebar)",
        },
    }

    @classmethod
    def get_theme_colors(cls, theme_name: str) -> Dict[str, str]:
        """Get color scheme for a specific theme."""
        return cls.THEMES.get(theme_name, cls.THEMES["light"])

    @classmethod
    def get_css_variables(cls, theme_name: str) -> str:
        """Generate CSS variables for a theme."""
        colors = cls.get_theme_colors(theme_name)
        css_vars = []

        for key, value in colors.items():
            css_key = key.replace("_", "-")
            css_vars.append(f"--{css_key}: {value};")

        return "\n".join(css_vars)

    @classmethod
    def get_theme_preview(cls, theme_name: str) -> Dict[str, Any]:
        """Get theme preview information."""
        colors = cls.get_theme_colors(theme_name)
        return {
            "name": theme_name,
            "colors": colors,
            "description": f"{theme_name.title()} theme with modern styling",
            "preview_url": f"/static/themes/{theme_name}_preview.png",
        }


class SystemMonitor:
    """Monitors system resources and performance."""

    @classmethod
    def get_system_info(cls) -> Dict[str, Any]:
        """Get comprehensive system information."""
        try:
            return {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "architecture": platform.architecture()[0],
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "hostname": platform.node(),
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}

    @classmethod
    def get_memory_info(cls) -> Dict[str, Any]:
        """Get memory usage information."""
        try:
            memory = psutil.virtual_memory()
            return {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "percent_used": memory.percent,
                "percent_available": 100 - memory.percent,
            }
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return {}

    @classmethod
    def get_cpu_info(cls) -> Dict[str, Any]:
        """Get CPU usage information."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()

            return {
                "usage_percent": cpu_percent,
                "core_count": cpu_count,
                "frequency_mhz": round(cpu_freq.current, 2) if cpu_freq else 0,
                "max_frequency_mhz": round(cpu_freq.max, 2) if cpu_freq else 0,
            }
        except Exception as e:
            logger.error(f"Error getting CPU info: {e}")
            return {}

    @classmethod
    def get_disk_info(cls) -> Dict[str, Any]:
        """Get disk usage information."""
        try:
            disk = psutil.disk_usage("/")
            return {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent_used": round((disk.used / disk.total) * 100, 2),
            }
        except Exception as e:
            logger.error(f"Error getting disk info: {e}")
            return {}

    @classmethod
    def get_network_info(cls) -> Dict[str, Any]:
        """Get network interface information."""
        try:
            network = psutil.net_io_counters()
            return {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv,
            }
        except Exception as e:
            logger.error(f"Error getting network info: {e}")
            return {}

    @classmethod
    def get_system_health(cls) -> Dict[str, Any]:
        """Get overall system health assessment."""
        try:
            memory = cls.get_memory_info()
            cpu = cls.get_cpu_info()
            disk = cls.get_disk_info()

            # Calculate health scores (0-100, higher is better)
            memory_score = 100 - memory.get("percent_used", 0)
            cpu_score = 100 - cpu.get("usage_percent", 0)
            disk_score = 100 - disk.get("percent_used", 0)

            overall_score = (memory_score + cpu_score + disk_score) / 3

            # Determine health status
            if overall_score >= 80:
                status = "excellent"
                color = "success"
            elif overall_score >= 60:
                status = "good"
                color = "info"
            elif overall_score >= 40:
                status = "fair"
                color = "warning"
            else:
                status = "poor"
                color = "danger"

            return {
                "overall_score": round(overall_score, 1),
                "status": status,
                "color": color,
                "memory_score": round(memory_score, 1),
                "cpu_score": round(cpu_score, 1),
                "disk_score": round(disk_score, 1),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {}


class SettingsValidator:
    """Validates settings and provides recommendations."""

    @classmethod
    def validate_general_settings(cls, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Validate general settings and provide recommendations."""
        errors = []
        warnings = []
        recommendations = []

        # Validate theme
        theme = settings.get("theme", "light")
        if theme not in ["light", "dark", "auto"]:
            errors.append("Invalid theme selection")

        # Validate page size
        page_size = settings.get("page_size", 25)
        if page_size not in [10, 25, 50, 100]:
            errors.append("Invalid page size")

        # Validate auto-refresh interval
        auto_refresh = settings.get("auto_refresh_interval", 30)
        if auto_refresh < 0:
            errors.append("Auto-refresh interval must be non-negative")
        elif auto_refresh < 5:
            warnings.append("Very short auto-refresh intervals may impact performance")
        elif auto_refresh > 300:
            warnings.append(
                "Very long auto-refresh intervals may reduce real-time updates"
            )

        # Recommendations
        if auto_refresh == 0:
            recommendations.append(
                "Consider enabling auto-refresh for better user experience"
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "recommendations": recommendations,
        }

    @classmethod
    def validate_analysis_settings(cls, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Validate analysis settings and provide recommendations."""
        errors = []
        warnings = []
        recommendations = []

        # Validate sampling frequency
        sampling_freq = settings.get("default_sampling_freq", 1000)
        if sampling_freq < 100:
            errors.append("Sampling frequency must be at least 100 Hz")
        elif sampling_freq < 500:
            warnings.append("Low sampling frequency may limit analysis accuracy")

        # Validate FFT points
        fft_points = settings.get("default_fft_points", 1024)
        if fft_points < 256:
            errors.append("FFT points must be at least 256")
        elif fft_points > 8192:
            warnings.append("Very high FFT points may impact performance")

        # Validate peak threshold
        peak_threshold = settings.get("peak_threshold", 0.5)
        if peak_threshold < 0.1 or peak_threshold > 1.0:
            errors.append("Peak threshold must be between 0.1 and 1.0")

        # Recommendations
        if fft_points < 1024:
            recommendations.append(
                "Consider increasing FFT points for better frequency resolution"
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "recommendations": recommendations,
        }

    @classmethod
    def validate_data_settings(cls, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data settings and provide recommendations."""
        errors = []
        warnings = []
        recommendations = []

        # Validate file size
        max_file_size = settings.get("max_file_size", 100)
        if max_file_size < 10 or max_file_size > 1000:
            errors.append("Max file size must be between 10 and 1000 MB")

        # Validate auto-save interval
        auto_save = settings.get("auto_save_interval", 5)
        if auto_save < 1 or auto_save > 60:
            errors.append("Auto-save interval must be between 1 and 60 minutes")

        # Validate retention period
        retention = settings.get("data_retention_days", 30)
        if retention < 1 or retention > 365:
            errors.append("Data retention must be between 1 and 365 days")

        # Recommendations
        if auto_save > 30:
            recommendations.append(
                "Consider shorter auto-save intervals for data safety"
            )

        if retention < 7:
            recommendations.append("Very short retention periods may lead to data loss")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "recommendations": recommendations,
        }

    @classmethod
    def validate_system_settings(cls, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Validate system settings and provide recommendations."""
        errors = []
        warnings = []
        recommendations = []

        # Validate CPU usage
        cpu_usage = settings.get("max_cpu_usage", 80)
        if cpu_usage < 10 or cpu_usage > 100:
            errors.append("Max CPU usage must be between 10% and 100%")

        # Validate memory limit
        memory_limit = settings.get("memory_limit_gb", 4)
        if memory_limit < 1 or memory_limit > 32:
            errors.append("Memory limit must be between 1 and 32 GB")

        # Validate parallel threads
        parallel_threads = settings.get("parallel_threads", 4)
        if parallel_threads < 1 or parallel_threads > 16:
            errors.append("Parallel threads must be between 1 and 16")

        # Get system info for recommendations
        system_info = SystemMonitor.get_system_info()
        cpu_count = SystemMonitor.get_cpu_info().get("core_count", 4)

        # Recommendations
        if parallel_threads > cpu_count:
            recommendations.append(
                f"Consider reducing parallel threads to {cpu_count} (available cores)"
            )

        if memory_limit > 8:
            warnings.append("High memory limits may impact system stability")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "recommendations": recommendations,
        }


class SettingsExporter:
    """Handles settings export and import operations."""

    @classmethod
    def export_settings_json(
        cls, settings: Dict[str, Any], filename: str = None
    ) -> str:
        """Export settings to JSON format."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vitaldsp_settings_{timestamp}.json"

        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "application": "vitalDSP Webapp",
            },
            "settings": settings,
        }

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Settings exported to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error exporting settings: {e}")
            raise

    @classmethod
    def import_settings_json(cls, filename: str) -> Dict[str, Any]:
        """Import settings from JSON format."""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                import_data = json.load(f)

            # Validate import data structure
            if "settings" not in import_data:
                raise ValueError("Invalid settings file format")

            logger.info(f"Settings imported from {filename}")
            return import_data["settings"]
        except Exception as e:
            logger.error(f"Error importing settings: {e}")
            raise


def get_system_recommendations() -> Dict[str, Any]:
    """Get system-specific recommendations for settings."""
    try:
        system_info = SystemMonitor.get_system_info()
        memory_info = SystemMonitor.get_memory_info()
        cpu_info = SystemMonitor.get_cpu_info()

        recommendations = {"general": [], "analysis": [], "data": [], "system": []}

        # System-specific recommendations
        if system_info.get("platform") == "Windows":
            recommendations["system"].append(
                "Windows: Consider enabling Windows Defender exclusions for upload folder"
            )

        if memory_info.get("total_gb", 0) < 4:
            recommendations["system"].append(
                "Low memory system: Consider reducing memory limits and parallel processing"
            )

        if cpu_info.get("core_count", 0) < 4:
            recommendations["analysis"].append(
                "Limited CPU cores: Consider reducing parallel processing threads"
            )

        return recommendations
    except Exception as e:
        logger.error(f"Error getting system recommendations: {e}")
        return {}


# Settings management functions for compatibility with tests
def load_user_settings() -> Dict[str, Any]:
    """Load user settings from file or return defaults."""
    try:
        settings_file = "settings.json"
        if os.path.exists(settings_file):
            with open(settings_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return get_default_settings()
    except Exception as e:
        logger.error(f"Error loading user settings: {e}")
        return get_default_settings()


def save_user_settings(settings: Dict[str, Any]) -> None:
    """Save user settings to file."""
    try:
        settings_file = "settings.json"
        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        logger.info("User settings saved successfully")
    except Exception as e:
        logger.error(f"Error saving user settings: {e}")
        raise


def get_setting_value(key: str, default: Any = None) -> Any:
    """Get a specific setting value."""
    try:
        settings = load_user_settings()
        return settings.get(key, default)
    except Exception as e:
        logger.error(f"Error getting setting value for {key}: {e}")
        return default


def set_setting_value(key: str, value: Any) -> None:
    """Set a specific setting value."""
    try:
        settings = load_user_settings()
        settings[key] = value
        save_user_settings(settings)
    except Exception as e:
        logger.error(f"Error setting value for {key}: {e}")
        raise


def reset_to_defaults() -> Dict[str, Any]:
    """Reset settings to default values."""
    try:
        default_settings = get_default_settings()
        save_user_settings(default_settings)
        return default_settings
    except Exception as e:
        logger.error(f"Error resetting to defaults: {e}")
        return get_default_settings()


def validate_setting_value(key: str, value: Any) -> bool:
    """Validate a specific setting value."""
    try:
        settings = {key: value}
        
        # Validate based on setting type
        if key in ["theme"]:
            return SettingsValidator.validate_general_settings(settings)["valid"]
        elif key in ["default_sampling_freq", "default_fft_points", "peak_threshold"]:
            return SettingsValidator.validate_analysis_settings(settings)["valid"]
        elif key in ["max_file_size", "auto_save_interval", "data_retention_days"]:
            return SettingsValidator.validate_data_settings(settings)["valid"]
        elif key in ["max_cpu_usage", "memory_limit_gb", "parallel_threads"]:
            return SettingsValidator.validate_system_settings(settings)["valid"]
        else:
            return True  # Unknown keys are considered valid
    except Exception as e:
        logger.error(f"Error validating setting {key}: {e}")
        return False


def get_setting_schema() -> Dict[str, Any]:
    """Get the settings schema/definition."""
    return {
        "theme": {
            "type": "string",
            "options": ["light", "dark", "auto"],
            "default": "light",
            "description": "Application theme"
        },
        "page_size": {
            "type": "integer",
            "options": [10, 25, 50, 100],
            "default": 25,
            "description": "Number of items per page"
        },
        "auto_refresh_interval": {
            "type": "integer",
            "min": 0,
            "max": 300,
            "default": 30,
            "description": "Auto-refresh interval in seconds"
        },
        "default_sampling_freq": {
            "type": "integer",
            "min": 100,
            "default": 1000,
            "description": "Default sampling frequency in Hz"
        },
        "default_fft_points": {
            "type": "integer",
            "min": 256,
            "default": 1024,
            "description": "Default FFT points"
        },
        "peak_threshold": {
            "type": "float",
            "min": 0.1,
            "max": 1.0,
            "default": 0.5,
            "description": "Peak detection threshold"
        },
        "max_file_size": {
            "type": "integer",
            "min": 10,
            "max": 1000,
            "default": 100,
            "description": "Maximum file size in MB"
        },
        "auto_save_interval": {
            "type": "integer",
            "min": 1,
            "max": 60,
            "default": 5,
            "description": "Auto-save interval in minutes"
        },
        "data_retention_days": {
            "type": "integer",
            "min": 1,
            "max": 365,
            "default": 30,
            "description": "Data retention period in days"
        },
        "max_cpu_usage": {
            "type": "integer",
            "min": 10,
            "max": 100,
            "default": 80,
            "description": "Maximum CPU usage percentage"
        },
        "memory_limit_gb": {
            "type": "integer",
            "min": 1,
            "max": 32,
            "default": 4,
            "description": "Memory limit in GB"
        },
        "parallel_threads": {
            "type": "integer",
            "min": 1,
            "max": 16,
            "default": 4,
            "description": "Number of parallel processing threads"
        }
    }


def apply_setting_constraints(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Apply constraints and validation to settings."""
    try:
        constrained_settings = settings.copy()
        
        # Apply general constraints
        general_validation = SettingsValidator.validate_general_settings(constrained_settings)
        if not general_validation["valid"]:
            logger.warning(f"General settings validation failed: {general_validation['errors']}")
        
        # Apply analysis constraints
        analysis_validation = SettingsValidator.validate_analysis_settings(constrained_settings)
        if not analysis_validation["valid"]:
            logger.warning(f"Analysis settings validation failed: {analysis_validation['errors']}")
        
        # Apply data constraints
        data_validation = SettingsValidator.validate_data_settings(constrained_settings)
        if not data_validation["valid"]:
            logger.warning(f"Data settings validation failed: {data_validation['errors']}")
        
        # Apply system constraints
        system_validation = SettingsValidator.validate_system_settings(constrained_settings)
        if not system_validation["valid"]:
            logger.warning(f"System settings validation failed: {system_validation['errors']}")
        
        return constrained_settings
    except Exception as e:
        logger.error(f"Error applying setting constraints: {e}")
        return settings


def export_settings(filename: str = None) -> str:
    """Export settings to a file."""
    try:
        settings = load_user_settings()
        return SettingsExporter.export_settings_json(settings, filename)
    except Exception as e:
        logger.error(f"Error exporting settings: {e}")
        raise


def import_settings(filename: str) -> Dict[str, Any]:
    """Import settings from a file."""
    try:
        return SettingsExporter.import_settings_json(filename)
    except Exception as e:
        logger.error(f"Error importing settings: {e}")
        raise


def backup_settings() -> str:
    """Create a backup of current settings."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"settings_backup_{timestamp}.json"
        settings = load_user_settings()
        return SettingsExporter.export_settings_json(settings, backup_filename)
    except Exception as e:
        logger.error(f"Error creating settings backup: {e}")
        raise


def restore_settings(backup_name: str) -> Dict[str, Any]:
    """Restore settings from a backup."""
    try:
        restored_settings = SettingsExporter.import_settings_json(backup_name)
        save_user_settings(restored_settings)
        return restored_settings
    except Exception as e:
        logger.error(f"Error restoring settings from {backup_name}: {e}")
        raise


def get_default_settings() -> Dict[str, Any]:
    """Get default settings values."""
    return {
        "theme": "light",
        "page_size": 25,
        "auto_refresh_interval": 30,
        "default_sampling_freq": 1000,
        "default_fft_points": 1024,
        "peak_threshold": 0.5,
        "max_file_size": 100,
        "auto_save_interval": 5,
        "data_retention_days": 30,
        "max_cpu_usage": 80,
        "memory_limit_gb": 4,
        "parallel_threads": 4
    }