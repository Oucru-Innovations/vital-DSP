"""
Comprehensive tests for vitalDSP_webapp.config.settings module.
Tests configuration classes, default values, and configuration management.
"""

import os
import tempfile
import shutil
from unittest.mock import patch
import pytest

# Import the modules we need to test
try:
    from vitalDSP_webapp.config.settings import (
        AppConfig,
        ColumnMapping,
        UIStyles,
        app_config,
        column_mapping,
        ui_styles,
        get_config,
        update_config
    )
except ImportError:
    # Fallback: add src to path if import fails
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..')
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from vitalDSP_webapp.config.settings import (
        AppConfig,
        ColumnMapping,
        UIStyles,
        app_config,
        column_mapping,
        ui_styles,
        get_config,
        update_config
    )


class TestAppConfig:
    """Test AppConfig class functionality"""
    
    def test_app_config_defaults(self):
        """Test AppConfig default values"""
        config = AppConfig()
        
        assert config.APP_NAME == "Vital-DSP Comprehensive Dashboard"
        assert config.APP_VERSION == "1.0.0"
        assert config.APP_DESCRIPTION == "Digital Signal Processing for Vital Signs"
        assert config.HOST == "0.0.0.0"
        assert config.PORT == 8000
        assert config.DEBUG is False
        assert config.MAX_FILE_SIZE == 100 * 1024 * 1024
        assert config.UPLOAD_FOLDER == "uploads"
        assert config.DEFAULT_SAMPLING_FREQ == 1000
        assert config.DEFAULT_TIME_UNIT == "ms"
        assert config.MAX_PREVIEW_ROWS == 10
        assert config.MAX_PLOT_POINTS == 1000
        assert config.THEME == "light"
        assert config.SIDEBAR_WIDTH == 280
        assert config.SIDEBAR_COLLAPSED_WIDTH == 70
        assert config.HEADER_HEIGHT == 64
        
    def test_app_config_signal_quality_thresholds(self):
        """Test signal quality threshold defaults"""
        config = AppConfig()
        
        assert config.SNR_EXCELLENT_THRESHOLD == 20.0
        assert config.SNR_GOOD_THRESHOLD == 15.0
        assert config.SNR_FAIR_THRESHOLD == 10.0
        assert config.ARTIFACT_EXCELLENT_RATIO == 0.01
        assert config.ARTIFACT_GOOD_RATIO == 0.05
        assert config.ARTIFACT_FAIR_RATIO == 0.1
        
    def test_app_config_post_init_allowed_extensions(self):
        """Test __post_init__ sets default allowed extensions"""
        config = AppConfig()
        
        expected_extensions = ['.csv', '.xlsx', '.xls', '.txt', '.dat']
        assert config.ALLOWED_EXTENSIONS == expected_extensions
        
    def test_app_config_post_init_custom_extensions(self):
        """Test __post_init__ preserves custom allowed extensions"""
        custom_extensions = ['.json', '.xml']
        config = AppConfig(ALLOWED_EXTENSIONS=custom_extensions)
        
        assert config.ALLOWED_EXTENSIONS == custom_extensions
        
    @patch('os.makedirs')
    def test_app_config_post_init_creates_upload_folder(self, mock_makedirs):
        """Test __post_init__ creates upload folder"""
        config = AppConfig(UPLOAD_FOLDER="test_uploads")
        
        mock_makedirs.assert_called_once_with("test_uploads", exist_ok=True)
        
    def test_app_config_custom_values(self):
        """Test AppConfig with custom values"""
        config = AppConfig(
            APP_NAME="Custom App",
            PORT=9000,
            DEBUG=True,
            THEME="dark"
        )
        
        assert config.APP_NAME == "Custom App"
        assert config.PORT == 9000
        assert config.DEBUG is True
        assert config.THEME == "dark"


class TestColumnMapping:
    """Test ColumnMapping class functionality"""
    
    def test_column_mapping_defaults(self):
        """Test ColumnMapping default values after __post_init__"""
        mapping = ColumnMapping()
        
        expected_time_patterns = ['time', 'timestamp', 'ms', 's', 'min', 'seconds']
        expected_signal_patterns = ['pleth', 'signal', 'waveform', 'ppg', 'ecg', 'amplitude']
        expected_ppg_patterns = ['pleth', 'ppg', 'photoplethysmography']
        expected_red_patterns = ['red', 'red_adc', 'red_adc_value']
        expected_ir_patterns = ['ir', 'ir_adc', 'ir_adc_value']
        expected_waveform_patterns = ['waveform', 'pleth', 'plethysmography', 'signal']
        expected_ecg_patterns = ['ecg', 'electrocardiogram', 'lead']
        
        assert mapping.TIME_PATTERNS == expected_time_patterns
        assert mapping.SIGNAL_PATTERNS == expected_signal_patterns
        assert mapping.PPG_PATTERNS == expected_ppg_patterns
        assert mapping.RED_PATTERNS == expected_red_patterns
        assert mapping.IR_PATTERNS == expected_ir_patterns
        assert mapping.WAVEFORM_PATTERNS == expected_waveform_patterns
        assert mapping.ECG_PATTERNS == expected_ecg_patterns
        
    def test_column_mapping_custom_patterns(self):
        """Test ColumnMapping with custom patterns"""
        custom_time = ['custom_time']
        custom_signal = ['custom_signal']
        
        mapping = ColumnMapping(
            TIME_PATTERNS=custom_time,
            SIGNAL_PATTERNS=custom_signal
        )
        
        assert mapping.TIME_PATTERNS == custom_time
        assert mapping.SIGNAL_PATTERNS == custom_signal
        # Other patterns should still get defaults
        assert mapping.PPG_PATTERNS == ['pleth', 'ppg', 'photoplethysmography']
        
    def test_column_mapping_partial_custom(self):
        """Test ColumnMapping with some custom and some default patterns"""
        custom_ppg = ['custom_ppg']
        
        mapping = ColumnMapping(PPG_PATTERNS=custom_ppg)
        
        # Custom pattern preserved
        assert mapping.PPG_PATTERNS == custom_ppg
        # Defaults still applied for others
        assert mapping.TIME_PATTERNS == ['time', 'timestamp', 'ms', 's', 'min', 'seconds']
        assert mapping.ECG_PATTERNS == ['ecg', 'electrocardiogram', 'lead']


class TestUIStyles:
    """Test UIStyles class functionality"""
    
    def test_ui_styles_defaults(self):
        """Test UIStyles default values"""
        styles = UIStyles()
        
        assert styles.PRIMARY_COLOR == "#3498db"
        assert styles.SUCCESS_COLOR == "#28a745"
        assert styles.WARNING_COLOR == "#ffc107"
        assert styles.DANGER_COLOR == "#dc3545"
        assert styles.INFO_COLOR == "#17a2b8"
        assert styles.CARD_PADDING == "1.5rem"
        assert styles.SECTION_MARGIN == "2rem"
        assert styles.COMPONENT_SPACING == "1rem"
        assert styles.UPLOAD_AREA_HEIGHT == "200px"
        assert styles.UPLOAD_BORDER_STYLE == "dashed"
        assert styles.UPLOAD_BORDER_RADIUS == "10px"
        assert styles.TABLE_STRIPED is True
        assert styles.TABLE_BORDERED is True
        assert styles.TABLE_HOVER is True
        assert styles.TABLE_SIZE == "sm"
        
    def test_ui_styles_custom_values(self):
        """Test UIStyles with custom values"""
        styles = UIStyles(
            PRIMARY_COLOR="#ff0000",
            TABLE_STRIPED=False,
            UPLOAD_AREA_HEIGHT="300px"
        )
        
        assert styles.PRIMARY_COLOR == "#ff0000"
        assert styles.TABLE_STRIPED is False
        assert styles.UPLOAD_AREA_HEIGHT == "300px"
        # Other defaults preserved
        assert styles.SUCCESS_COLOR == "#28a745"
        assert styles.TABLE_BORDERED is True


class TestGlobalInstances:
    """Test global configuration instances"""
    
    def test_global_app_config_exists(self):
        """Test global app_config instance exists and is AppConfig"""
        assert isinstance(app_config, AppConfig)
        assert app_config.APP_NAME == "Vital-DSP Comprehensive Dashboard"
        
    def test_global_column_mapping_exists(self):
        """Test global column_mapping instance exists and is ColumnMapping"""
        assert isinstance(column_mapping, ColumnMapping)
        assert 'time' in column_mapping.TIME_PATTERNS
        
    def test_global_ui_styles_exists(self):
        """Test global ui_styles instance exists and is UIStyles"""
        assert isinstance(ui_styles, UIStyles)
        assert ui_styles.PRIMARY_COLOR == "#3498db"


class TestConfigurationFunctions:
    """Test configuration management functions"""
    
    def test_get_config_returns_dict(self):
        """Test get_config returns dictionary with all configs"""
        config_dict = get_config()
        
        assert isinstance(config_dict, dict)
        assert 'app' in config_dict
        assert 'columns' in config_dict
        assert 'ui' in config_dict
        assert isinstance(config_dict['app'], AppConfig)
        assert isinstance(config_dict['columns'], ColumnMapping)
        assert isinstance(config_dict['ui'], UIStyles)
        
    def test_update_config_app_config(self):
        """Test update_config updates app_config attributes"""
        original_port = app_config.PORT
        
        update_config(port=9999)
        
        assert app_config.PORT == 9999
        
        # Restore original value
        app_config.PORT = original_port
        
    def test_update_config_column_mapping(self):
        """Test update_config updates column_mapping attributes"""
        original_patterns = column_mapping.TIME_PATTERNS[:]
        
        update_config(time_patterns=['custom_time'])
        
        assert column_mapping.TIME_PATTERNS == ['custom_time']
        
        # Restore original value
        column_mapping.TIME_PATTERNS = original_patterns
        
    def test_update_config_ui_styles(self):
        """Test update_config updates ui_styles attributes"""
        original_color = ui_styles.PRIMARY_COLOR
        
        update_config(primary_color="#ff0000")
        
        assert ui_styles.PRIMARY_COLOR == "#ff0000"
        
        # Restore original value
        ui_styles.PRIMARY_COLOR = original_color
        
    def test_update_config_invalid_attribute(self):
        """Test update_config ignores invalid attributes"""
        # This should not raise an error
        update_config(invalid_attribute="value")
        
        # No assertion needed, just testing it doesn't crash
        
    def test_update_config_mixed_attributes(self):
        """Test update_config with attributes from different configs"""
        original_port = app_config.PORT
        original_color = ui_styles.PRIMARY_COLOR
        
        update_config(
            port=8888,
            primary_color="#00ff00"
        )
        
        assert app_config.PORT == 8888
        assert ui_styles.PRIMARY_COLOR == "#00ff00"
        
        # Restore original values
        app_config.PORT = original_port
        ui_styles.PRIMARY_COLOR = original_color


class TestConfigurationIntegration:
    """Test configuration integration scenarios"""
    
    def test_config_instances_are_independent(self):
        """Test that creating new instances doesn't affect globals"""
        new_config = AppConfig(PORT=7777)
        
        assert new_config.PORT == 7777
        assert app_config.PORT != 7777  # Global should be unchanged
        
    def test_upload_folder_creation_integration(self):
        """Test upload folder creation in realistic scenario"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_upload_path = os.path.join(temp_dir, "test_uploads")
            
            config = AppConfig(UPLOAD_FOLDER=test_upload_path)
            
            assert os.path.exists(test_upload_path)
            assert os.path.isdir(test_upload_path)
            
    def test_configuration_serialization(self):
        """Test that configuration can be serialized/accessed"""
        config_dict = get_config()
        
        # Test accessing nested attributes
        assert config_dict['app'].APP_NAME == "Vital-DSP Comprehensive Dashboard"
        assert 'time' in config_dict['columns'].TIME_PATTERNS
        assert config_dict['ui'].PRIMARY_COLOR.startswith('#')
        
    def test_pattern_matching_scenarios(self):
        """Test column mapping patterns for realistic scenarios"""
        mapping = ColumnMapping()
        
        # Test that common column names would match
        time_columns = ['time', 'timestamp', 'seconds', 'ms']
        for col in time_columns:
            assert col in mapping.TIME_PATTERNS
            
        signal_columns = ['signal', 'ppg', 'ecg', 'waveform']
        for col in signal_columns:
            assert col in mapping.SIGNAL_PATTERNS
            
    def test_theme_and_styling_consistency(self):
        """Test UI theme and styling consistency"""
        styles = UIStyles()
        
        # Test color format consistency
        colors = [
            styles.PRIMARY_COLOR,
            styles.SUCCESS_COLOR,
            styles.WARNING_COLOR,
            styles.DANGER_COLOR,
            styles.INFO_COLOR
        ]
        
        for color in colors:
            assert color.startswith('#')
            assert len(color) == 7  # #RRGGBB format
            
        # Test size format consistency
        sizes = [
            styles.CARD_PADDING,
            styles.SECTION_MARGIN,
            styles.COMPONENT_SPACING,
            styles.UPLOAD_AREA_HEIGHT
        ]
        
        for size in sizes:
            assert isinstance(size, str)
            assert any(unit in size for unit in ['px', 'rem', 'em', '%'])
