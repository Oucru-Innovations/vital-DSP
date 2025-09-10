"""
Complete tests for vitalDSP_webapp.utils.settings_utils module.
Tests all classes and functions to improve coverage.
"""

import pytest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime

# Import the modules we need to test
from vitalDSP_webapp.utils.settings_utils import (
    ThemeManager,
    SystemMonitor,
    SettingsValidator,
    SettingsExporter,
    get_system_recommendations
)
SETTINGS_UTILS_AVAILABLE = True


@pytest.mark.skipif(not SETTINGS_UTILS_AVAILABLE, reason="Settings utils module not available")
class TestThemeManager:
    """Test ThemeManager class"""
    
    def test_theme_manager_has_themes(self):
        """Test that ThemeManager has predefined themes"""
        assert hasattr(ThemeManager, 'THEMES')
        assert isinstance(ThemeManager.THEMES, dict)
        assert 'light' in ThemeManager.THEMES
        assert 'dark' in ThemeManager.THEMES
        assert 'auto' in ThemeManager.THEMES
        
    def test_get_theme_colors_light(self):
        """Test getting light theme colors"""
        colors = ThemeManager.get_theme_colors('light')
        
        assert isinstance(colors, dict)
        assert 'primary_color' in colors
        assert 'background_color' in colors
        assert 'text_color' in colors
        assert colors['background_color'] == '#ffffff'
        assert colors['text_color'] == '#2c3e50'
        
    def test_get_theme_colors_dark(self):
        """Test getting dark theme colors"""
        colors = ThemeManager.get_theme_colors('dark')
        
        assert isinstance(colors, dict)
        assert 'primary_color' in colors
        assert 'background_color' in colors
        assert 'text_color' in colors
        assert colors['background_color'] == '#1a1a1a'
        assert colors['text_color'] == '#ffffff'
        
    def test_get_theme_colors_auto(self):
        """Test getting auto theme colors"""
        colors = ThemeManager.get_theme_colors('auto')
        
        assert isinstance(colors, dict)
        assert 'primary_color' in colors
        assert 'background_color' in colors
        assert 'text_color' in colors
        assert 'var(--system-background)' in colors['background_color']
        
    def test_get_theme_colors_invalid(self):
        """Test getting colors for invalid theme"""
        colors = ThemeManager.get_theme_colors('invalid_theme')
        
        # Should return light theme as default
        light_colors = ThemeManager.get_theme_colors('light')
        assert colors == light_colors
        
    def test_theme_structure_consistency(self):
        """Test that all themes have consistent structure"""
        required_keys = {
            'primary_color', 'secondary_color', 'success_color', 'warning_color',
            'danger_color', 'info_color', 'background_color', 'text_color',
            'border_color', 'card_background', 'sidebar_background'
        }
        
        for theme_name, theme_data in ThemeManager.THEMES.items():
            assert set(theme_data.keys()) == required_keys, f"Theme {theme_name} missing keys"


@pytest.mark.skipif(not SETTINGS_UTILS_AVAILABLE, reason="Settings utils module not available")
class TestSystemMonitor:
    """Test SystemMonitor class"""
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_get_system_info(self, mock_disk, mock_memory, mock_cpu):
        """Test getting system information"""
        # Mock system data
        mock_cpu.return_value = 45.2
        mock_memory.return_value = MagicMock(total=8589934592, available=4294967296, percent=50.0)
        mock_disk.return_value = MagicMock(total=1000000000000, free=500000000000, percent=50.0)
        
        system_info = SystemMonitor.get_system_info()
        
        assert isinstance(system_info, dict)
        assert 'platform' in system_info
        # Check for actual keys returned by get_system_info
        assert 'platform' in system_info
        assert 'hostname' in system_info
        assert 'architecture' in system_info
        
    @patch('psutil.cpu_count')
    @patch('psutil.cpu_freq')
    def test_get_cpu_info(self, mock_cpu_freq, mock_cpu_count):
        """Test getting CPU information"""
        mock_cpu_count.return_value = 8
        mock_cpu_freq.return_value = MagicMock(current=3200.0, min=800.0, max=4000.0)
        
        cpu_info = SystemMonitor.get_cpu_info()
        
        assert isinstance(cpu_info, dict)
        assert 'core_count' in cpu_info
        assert 'frequency_mhz' in cpu_info
        
    @patch('psutil.virtual_memory')
    def test_get_memory_info(self, mock_memory):
        """Test getting memory information"""
        mock_memory.return_value = MagicMock(
            total=16777216000,
            available=8388608000,
            percent=50.0,
            used=8388608000
        )
        
        memory_info = SystemMonitor.get_memory_info()
        
        assert isinstance(memory_info, dict)
        assert 'total_gb' in memory_info
        assert 'available_gb' in memory_info
        assert 'percent_used' in memory_info
        
    @patch('psutil.disk_usage')
    def test_get_disk_info(self, mock_disk):
        """Test getting disk information"""
        mock_disk.return_value = MagicMock(
            total=2000000000000,
            free=1000000000000,
            used=1000000000000
        )
        
        disk_info = SystemMonitor.get_disk_info()
        
        assert isinstance(disk_info, dict)
        assert 'total_gb' in disk_info
        assert 'free_gb' in disk_info
        assert 'percent_used' in disk_info
        
    @patch.object(SystemMonitor, 'get_system_info')
    @patch.object(SystemMonitor, 'get_cpu_info')
    @patch.object(SystemMonitor, 'get_memory_info')
    @patch.object(SystemMonitor, 'get_disk_info')
    def test_get_system_health(self, mock_disk_info, mock_memory_info, mock_cpu_info, mock_system_info):
        """Test getting system health summary"""
        # Mock all the info methods
        mock_system_info.return_value = {'platform': 'Linux'}
        mock_cpu_info.return_value = {'core_count': 8, 'usage_percent': 45.0}
        mock_memory_info.return_value = {'percent_used': 60.0}
        mock_disk_info.return_value = {'percent_used': 70.0}
        
        health = SystemMonitor.get_system_health()
        
        assert isinstance(health, dict)
        assert 'status' in health
        assert 'overall_score' in health
        assert 'memory_score' in health
        assert 'cpu_score' in health
        assert 'disk_score' in health


@pytest.mark.skipif(not SETTINGS_UTILS_AVAILABLE, reason="Settings utils module not available")
class TestSettingsValidator:
    """Test SettingsValidator class"""
    
    def test_validate_general_settings_valid(self):
        """Test validating valid general settings"""
        settings = {
            'theme': 'dark',
            'page_size': 25,
            'auto_refresh_interval': 30
        }
        
        result = SettingsValidator.validate_general_settings(settings)
        
        assert isinstance(result, dict)
        assert result['valid'] is True
        assert len(result['errors']) == 0
        
    def test_validate_general_settings_invalid_theme(self):
        """Test validating invalid theme"""
        settings = {
            'theme': 'invalid_theme',
            'page_size': 25,
            'auto_refresh_interval': 30
        }
        
        result = SettingsValidator.validate_general_settings(settings)
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert any('theme' in error.lower() for error in result['errors'])
        
    def test_validate_general_settings_invalid_page_size(self):
        """Test validating invalid page size"""
        settings = {
            'theme': 'light',
            'page_size': 75,  # Invalid page size
            'auto_refresh_interval': 30
        }
        
        result = SettingsValidator.validate_general_settings(settings)
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert any('page size' in error.lower() for error in result['errors'])
        
    def test_validate_general_settings_negative_refresh(self):
        """Test validating negative refresh interval"""
        settings = {
            'theme': 'light',
            'page_size': 25,
            'auto_refresh_interval': -5
        }
        
        result = SettingsValidator.validate_general_settings(settings)
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
        
    def test_validate_general_settings_warnings(self):
        """Test settings that generate warnings"""
        settings = {
            'theme': 'light',
            'page_size': 25,
            'auto_refresh_interval': 2  # Very short interval
        }
        
        result = SettingsValidator.validate_general_settings(settings)
        
        assert isinstance(result, dict)
        assert 'warnings' in result
        assert len(result['warnings']) > 0
        
    def test_validate_analysis_settings_valid(self):
        """Test validating valid analysis settings"""
        settings = {
            'default_sampling_freq': 1000,
            'default_fft_points': 1024,
            'peak_threshold': 0.5
        }
        
        result = SettingsValidator.validate_analysis_settings(settings)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
        
    def test_validate_analysis_settings_low_sampling_freq(self):
        """Test validating low sampling frequency"""
        settings = {
            'default_sampling_freq': 50,  # Too low
            'default_fft_points': 1024,
            'peak_threshold': 0.5
        }
        
        result = SettingsValidator.validate_analysis_settings(settings)
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
        
    def test_validate_analysis_settings_invalid_fft_points(self):
        """Test validating invalid FFT points"""
        settings = {
            'default_sampling_freq': 1000,
            'default_fft_points': 128,  # Too low
            'peak_threshold': 0.5
        }
        
        result = SettingsValidator.validate_analysis_settings(settings)
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
        
    def test_validate_analysis_settings_invalid_peak_threshold(self):
        """Test validating invalid peak threshold"""
        settings = {
            'default_sampling_freq': 1000,
            'default_fft_points': 1024,
            'peak_threshold': 1.5  # Too high
        }
        
        result = SettingsValidator.validate_analysis_settings(settings)
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
        
    def test_validate_data_settings_valid(self):
        """Test validating valid data settings"""
        settings = {
            'max_file_size_mb': 50,
            'cache_size_mb': 100,
            'backup_enabled': True
        }
        
        result = SettingsValidator.validate_data_settings(settings)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
        
    @patch.object(SystemMonitor, 'get_cpu_info')
    @patch.object(SystemMonitor, 'get_system_info')
    def test_validate_system_settings_valid(self, mock_system_info, mock_cpu_info):
        """Test validating valid system settings"""
        mock_cpu_info.return_value = {'core_count': 8}
        mock_system_info.return_value = {'platform': 'Linux'}
        
        settings = {
            'max_cpu_usage': 80,
            'memory_limit_gb': 4,
            'parallel_threads': 4
        }
        
        result = SettingsValidator.validate_system_settings(settings)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
        
    @patch.object(SystemMonitor, 'get_cpu_info')
    @patch.object(SystemMonitor, 'get_system_info')
    def test_validate_system_settings_invalid_cpu(self, mock_system_info, mock_cpu_info):
        """Test validating invalid CPU usage"""
        mock_cpu_info.return_value = {'core_count': 8}
        mock_system_info.return_value = {'platform': 'Linux'}
        
        settings = {
            'max_cpu_usage': 150,  # Invalid
            'memory_limit_gb': 4,
            'parallel_threads': 4
        }
        
        result = SettingsValidator.validate_system_settings(settings)
        
        assert result['valid'] is False
        assert len(result['errors']) > 0


@pytest.mark.skipif(not SETTINGS_UTILS_AVAILABLE, reason="Settings utils module not available")
class TestSettingsExporter:
    """Test SettingsExporter class"""
    
    def test_export_settings_json_with_filename(self):
        """Test exporting settings to JSON with specified filename"""
        settings = {'theme': 'dark', 'page_size': 25}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name
            
        try:
            result_filename = SettingsExporter.export_settings_json(settings, filename)
            
            assert result_filename == filename
            assert os.path.exists(filename)
            
            # Verify file contents
            with open(filename, 'r') as f:
                data = json.load(f)
            
            assert 'export_info' in data
            assert 'settings' in data
            assert data['settings'] == settings
            
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
                
    def test_export_settings_json_auto_filename(self):
        """Test exporting settings with auto-generated filename"""
        settings = {'theme': 'light', 'page_size': 50}
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                filename = SettingsExporter.export_settings_json(settings)
                
                assert filename.startswith('vitaldsp_settings_')
                assert filename.endswith('.json')
                mock_file.assert_called_once()
                mock_json_dump.assert_called_once()
                
    def test_export_settings_json_error_handling(self):
        """Test error handling during export"""
        settings = {'theme': 'dark'}
        
        with patch('builtins.open', side_effect=IOError("Permission denied")):
            with pytest.raises(IOError):
                SettingsExporter.export_settings_json(settings, 'test.json')
                
    def test_import_settings_json_valid_file(self):
        """Test importing settings from valid JSON file"""
        settings = {'theme': 'dark', 'page_size': 25}
        export_data = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'application': 'vitalDSP Webapp'
            },
            'settings': settings
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(export_data, f)
            filename = f.name
            
        try:
            imported_settings = SettingsExporter.import_settings_json(filename)
            
            assert imported_settings == settings
            
        finally:
            os.unlink(filename)
            
    def test_import_settings_json_invalid_format(self):
        """Test importing settings from invalid JSON format"""
        invalid_data = {'invalid': 'format'}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_data, f)
            filename = f.name
            
        try:
            with pytest.raises(ValueError, match="Invalid settings file format"):
                SettingsExporter.import_settings_json(filename)
                
        finally:
            os.unlink(filename)
            
    def test_import_settings_json_file_not_found(self):
        """Test importing settings from non-existent file"""
        with pytest.raises(FileNotFoundError):
            SettingsExporter.import_settings_json('non_existent_file.json')
            
    def test_import_settings_json_invalid_json(self):
        """Test importing settings from file with invalid JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('invalid json content')
            filename = f.name
            
        try:
            with pytest.raises(json.JSONDecodeError):
                SettingsExporter.import_settings_json(filename)
                
        finally:
            os.unlink(filename)


@pytest.mark.skipif(not SETTINGS_UTILS_AVAILABLE, reason="Settings utils module not available")
class TestSystemRecommendations:
    """Test get_system_recommendations function"""
    
    @patch.object(SystemMonitor, 'get_system_health')
    @patch.object(SystemMonitor, 'get_cpu_info')
    @patch.object(SystemMonitor, 'get_memory_info')
    def test_get_system_recommendations(self, mock_memory_info, mock_cpu_info, mock_system_health):
        """Test getting system recommendations"""
        mock_system_health.return_value = {
            'status': 'healthy',
            'system_info': {'cpu_usage': 30.0},
            'cpu_info': {'core_count': 8},
            'memory_info': {'usage_percent': 50.0},
            'disk_info': {'usage_percent': 60.0}
        }
        mock_cpu_info.return_value = {'core_count': 8}
        mock_memory_info.return_value = {'total_gb': 16}
        
        recommendations = get_system_recommendations()
        
        assert isinstance(recommendations, dict)
        assert 'general' in recommendations
        assert 'analysis' in recommendations
        assert 'data' in recommendations
        assert 'system' in recommendations


@pytest.mark.skipif(not SETTINGS_UTILS_AVAILABLE, reason="Settings utils module not available")
class TestIntegration:
    """Test integration between different components"""
    
    @patch.object(SystemMonitor, 'get_system_health')
    def test_theme_and_system_integration(self, mock_system_health):
        """Test integration between theme manager and system monitor"""
        mock_system_health.return_value = {'status': 'healthy'}
        
        # Get theme colors
        colors = ThemeManager.get_theme_colors('dark')
        
        # Get system health
        health = SystemMonitor.get_system_health()
        
        # Both should work independently
        assert isinstance(colors, dict)
        assert isinstance(health, dict)
        
    def test_validator_and_exporter_integration(self):
        """Test integration between validator and exporter"""
        settings = {
            'theme': 'dark',
            'page_size': 25,
            'auto_refresh_interval': 30
        }
        
        # Validate settings
        validation_result = SettingsValidator.validate_general_settings(settings)
        assert validation_result['valid'] is True
        
        # Export valid settings
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                filename = SettingsExporter.export_settings_json(settings)
                
                assert filename is not None
                mock_file.assert_called_once()
                
    @patch.object(SystemMonitor, 'get_cpu_info')
    def test_system_monitor_and_validator_integration(self, mock_cpu_info):
        """Test integration between system monitor and validator"""
        mock_cpu_info.return_value = {'core_count': 4}
        
        # Get CPU info
        cpu_info = SystemMonitor.get_cpu_info()
        
        # Use CPU info in system settings validation
        settings = {
            'max_cpu_usage': 80,
            'memory_limit_gb': 4,
            'parallel_threads': cpu_info['core_count']
        }
        
        with patch.object(SystemMonitor, 'get_system_info') as mock_system_info:
            mock_system_info.return_value = {'platform': 'Linux'}
            result = SettingsValidator.validate_system_settings(settings)
            
        assert result['valid'] is True


class TestSettingsUtilsBasic:
    """Basic tests that work even when module is not fully available"""
    
    def test_module_imports(self):
        """Test that we can import the module components"""
        if SETTINGS_UTILS_AVAILABLE:
            assert ThemeManager is not None
            assert SystemMonitor is not None
            assert SettingsValidator is not None
            assert SettingsExporter is not None
            assert get_system_recommendations is not None
        else:
            # If not available, just pass
            assert True
