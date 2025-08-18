"""
Configuration settings for vitalDSP webapp.
Centralized configuration management for all webapp components.
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class AppConfig:
    """Application configuration settings."""
    
    # App metadata
    APP_NAME: str = "Vital-DSP Comprehensive Dashboard"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Digital Signal Processing for Vital Signs"
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # File upload settings
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: List[str] = None
    UPLOAD_FOLDER: str = "uploads"
    
    # Data processing settings
    DEFAULT_SAMPLING_FREQ: int = 1000
    DEFAULT_TIME_UNIT: str = "ms"
    MAX_PREVIEW_ROWS: int = 10
    MAX_PLOT_POINTS: int = 1000
    
    # UI settings
    THEME: str = "light"
    SIDEBAR_WIDTH: int = 280
    SIDEBAR_COLLAPSED_WIDTH: int = 70
    HEADER_HEIGHT: int = 64
    
    # Signal quality thresholds
    SNR_EXCELLENT_THRESHOLD: float = 20.0
    SNR_GOOD_THRESHOLD: float = 15.0
    SNR_FAIR_THRESHOLD: float = 10.0
    ARTIFACT_EXCELLENT_RATIO: float = 0.01
    ARTIFACT_GOOD_RATIO: float = 0.05
    ARTIFACT_FAIR_RATIO: float = 0.1
    
    def __post_init__(self):
        """Set default values for complex types."""
        if self.ALLOWED_EXTENSIONS is None:
            self.ALLOWED_EXTENSIONS = ['.csv', '.xlsx', '.xls', '.txt', '.dat']
        
        # Ensure upload folder exists
        os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)


@dataclass
class ColumnMapping:
    """Column mapping configuration for different signal types."""
    
    # Time column patterns
    TIME_PATTERNS: List[str] = None
    
    # Signal column patterns
    SIGNAL_PATTERNS: List[str] = None
    
    # PPG-specific patterns
    PPG_PATTERNS: List[str] = None
    RED_PATTERNS: List[str] = None
    IR_PATTERNS: List[str] = None
    WAVEFORM_PATTERNS: List[str] = None
    
    # ECG-specific patterns
    ECG_PATTERNS: List[str] = None
    
    def __post_init__(self):
        """Set default values for pattern matching."""
        if self.TIME_PATTERNS is None:
            self.TIME_PATTERNS = ['time', 'timestamp', 'ms', 's', 'min', 'seconds']
        
        if self.SIGNAL_PATTERNS is None:
            self.SIGNAL_PATTERNS = ['pleth', 'signal', 'waveform', 'ppg', 'ecg', 'amplitude']
        
        if self.PPG_PATTERNS is None:
            self.PPG_PATTERNS = ['pleth', 'ppg', 'photoplethysmography']
        
        if self.RED_PATTERNS is None:
            self.RED_PATTERNS = ['red', 'red_adc', 'red_adc_value']
        
        if self.IR_PATTERNS is None:
            self.IR_PATTERNS = ['ir', 'ir_adc', 'ir_adc_value']
        
        if self.WAVEFORM_PATTERNS is None:
            self.WAVEFORM_PATTERNS = ['waveform', 'pleth', 'plethysmography', 'signal']
        
        if self.ECG_PATTERNS is None:
            self.ECG_PATTERNS = ['ecg', 'electrocardiogram', 'lead']


@dataclass
class UIStyles:
    """UI styling constants."""
    
    # Colors
    PRIMARY_COLOR: str = "#3498db"
    SUCCESS_COLOR: str = "#28a745"
    WARNING_COLOR: str = "#ffc107"
    DANGER_COLOR: str = "#dc3545"
    INFO_COLOR: str = "#17a2b8"
    
    # Spacing
    CARD_PADDING: str = "1.5rem"
    SECTION_MARGIN: str = "2rem"
    COMPONENT_SPACING: str = "1rem"
    
    # Upload area styling
    UPLOAD_AREA_HEIGHT: str = "200px"
    UPLOAD_BORDER_STYLE: str = "dashed"
    UPLOAD_BORDER_RADIUS: str = "10px"
    
    # Table styling
    TABLE_STRIPED: bool = True
    TABLE_BORDERED: bool = True
    TABLE_HOVER: bool = True
    TABLE_SIZE: str = "sm"


# Global configuration instances
app_config = AppConfig()
column_mapping = ColumnMapping()
ui_styles = UIStyles()


def get_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary."""
    return {
        'app': app_config,
        'columns': column_mapping,
        'ui': ui_styles
    }


def update_config(**kwargs) -> None:
    """Update configuration values."""
    for key, value in kwargs.items():
        if hasattr(app_config, key.upper()):
            setattr(app_config, key.upper(), value)
        elif hasattr(column_mapping, key.upper()):
            setattr(column_mapping, key.upper(), value)
        elif hasattr(ui_styles, key.upper()):
            setattr(ui_styles, key.upper(), value)
