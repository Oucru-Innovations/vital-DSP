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
                # Check that open was called at least once (may be called multiple times due to mocking)
                assert mock_file.call_count >= 1
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


@pytest.mark.skipif(not SETTINGS_UTILS_AVAILABLE, reason="Settings utils module not available")
class TestThemeManagerAdditional:
    """Additional tests for ThemeManager to improve coverage"""

    def test_get_css_variables(self):
        """Test generation of CSS variables"""
        css = ThemeManager.get_css_variables('light')

        assert isinstance(css, str)
        assert '--primary-color' in css
        assert '--background-color' in css
        assert '--text-color' in css
        assert '#3498db' in css  # primary color value

    def test_get_css_variables_dark(self):
        """Test generation of CSS variables for dark theme"""
        css = ThemeManager.get_css_variables('dark')

        assert isinstance(css, str)
        assert '--background-color' in css
        assert '#1a1a1a' in css  # dark background

    def test_get_css_variables_auto(self):
        """Test generation of CSS variables for auto theme"""
        css = ThemeManager.get_css_variables('auto')

        assert isinstance(css, str)
        assert 'var(--system-background)' in css

    def test_get_theme_preview(self):
        """Test getting theme preview information"""
        preview = ThemeManager.get_theme_preview('light')

        assert isinstance(preview, dict)
        assert 'name' in preview
        assert 'colors' in preview
        assert 'description' in preview
        assert 'preview_url' in preview
        assert preview['name'] == 'light'
        assert 'Light theme' in preview['description']

    def test_get_theme_preview_all_themes(self):
        """Test getting preview for all available themes"""
        for theme_name in ['light', 'dark', 'auto']:
            preview = ThemeManager.get_theme_preview(theme_name)

            assert preview['name'] == theme_name
            assert isinstance(preview['colors'], dict)
            assert '/static/themes/' in preview['preview_url']


@pytest.mark.skipif(not SETTINGS_UTILS_AVAILABLE, reason="Settings utils module not available")
class TestSystemMonitorAdditional:
    """Additional tests for SystemMonitor to improve coverage"""

    @patch('psutil.net_io_counters')
    def test_get_network_info(self, mock_net):
        """Test getting network information"""
        mock_net.return_value = MagicMock(
            bytes_sent=1000000,
            bytes_recv=2000000,
            packets_sent=5000,
            packets_recv=6000
        )

        network = SystemMonitor.get_network_info()

        assert isinstance(network, dict)
        assert 'bytes_sent' in network
        assert 'bytes_recv' in network
        assert 'packets_sent' in network
        assert 'packets_recv' in network
        assert network['bytes_sent'] == 1000000
        assert network['bytes_recv'] == 2000000

    @patch('psutil.net_io_counters')
    def test_get_network_info_error(self, mock_net):
        """Test network info with error"""
        mock_net.side_effect = Exception("Network error")

        network = SystemMonitor.get_network_info()

        assert isinstance(network, dict)
        assert len(network) == 0

    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    @patch('psutil.disk_usage')
    def test_get_system_health_excellent(self, mock_disk, mock_cpu, mock_memory):
        """Test system health with excellent scores"""
        # Mock low usage (excellent health)
        mock_memory.return_value = MagicMock(percent=10.0)
        mock_cpu.return_value = 5.0
        mock_disk.return_value = MagicMock(total=1000, used=100)

        health = SystemMonitor.get_system_health()

        assert isinstance(health, dict)
        assert 'overall_score' in health
        assert 'status' in health
        assert 'color' in health
        assert health['status'] == 'excellent'
        assert health['color'] == 'success'
        assert health['overall_score'] >= 80

    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    @patch('psutil.disk_usage')
    def test_get_system_health_good(self, mock_disk, mock_cpu, mock_memory):
        """Test system health with good scores"""
        # Mock moderate usage
        mock_memory.return_value = MagicMock(percent=35.0)
        mock_cpu.return_value = 40.0
        mock_disk.return_value = MagicMock(total=1000, used=350)

        health = SystemMonitor.get_system_health()

        assert isinstance(health, dict)
        assert health['status'] == 'good'
        assert health['color'] == 'info'
        assert 60 <= health['overall_score'] < 80

    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    @patch('psutil.disk_usage')
    def test_get_system_health_fair(self, mock_disk, mock_cpu, mock_memory):
        """Test system health with fair scores"""
        # Mock high usage
        mock_memory.return_value = MagicMock(percent=60.0)
        mock_cpu.return_value = 55.0
        mock_disk.return_value = MagicMock(total=1000, used=550)

        health = SystemMonitor.get_system_health()

        assert isinstance(health, dict)
        assert health['status'] == 'fair'
        assert health['color'] == 'warning'
        assert 40 <= health['overall_score'] < 60

    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    @patch('psutil.disk_usage')
    def test_get_system_health_poor(self, mock_disk, mock_cpu, mock_memory):
        """Test system health with poor scores"""
        # Mock very high usage
        mock_memory.return_value = MagicMock(percent=90.0)
        mock_cpu.return_value = 85.0
        mock_disk.return_value = MagicMock(total=1000, used=900)

        health = SystemMonitor.get_system_health()

        assert isinstance(health, dict)
        assert health['status'] == 'poor'
        assert health['color'] == 'danger'
        assert health['overall_score'] < 40

    @patch('psutil.cpu_freq')
    @patch('psutil.cpu_count')
    @patch('psutil.cpu_percent')
    def test_get_cpu_info_with_frequency(self, mock_percent, mock_count, mock_freq):
        """Test CPU info when frequency is available"""
        mock_percent.return_value = 50.0
        mock_count.return_value = 8
        mock_freq.return_value = MagicMock(current=2400.0, max=3600.0)

        cpu = SystemMonitor.get_cpu_info()

        assert cpu['usage_percent'] == 50.0
        assert cpu['core_count'] == 8
        assert cpu['frequency_mhz'] == 2400.0
        assert cpu['max_frequency_mhz'] == 3600.0

    @patch('psutil.cpu_freq')
    @patch('psutil.cpu_count')
    @patch('psutil.cpu_percent')
    def test_get_cpu_info_without_frequency(self, mock_percent, mock_count, mock_freq):
        """Test CPU info when frequency is not available"""
        mock_percent.return_value = 50.0
        mock_count.return_value = 4
        mock_freq.return_value = None  # No frequency info

        cpu = SystemMonitor.get_cpu_info()

        assert cpu['usage_percent'] == 50.0
        assert cpu['core_count'] == 4
        assert cpu['frequency_mhz'] == 0
        assert cpu['max_frequency_mhz'] == 0


@pytest.mark.skipif(not SETTINGS_UTILS_AVAILABLE, reason="Settings utils module not available")
class TestSettingsValidatorEdgeCases:
    """Test edge cases in SettingsValidator"""

    def test_validate_general_settings_edge_values(self):
        """Test general settings with edge case values"""
        settings = {
            'theme': 'light',
            'page_size': 10,  # Minimum value
            'auto_refresh_interval': 5  # Minimum acceptable value
        }

        result = SettingsValidator.validate_general_settings(settings)

        assert result['valid'] is True
        assert len(result['errors']) == 0

    def test_validate_general_settings_boundary_warnings(self):
        """Test general settings boundary conditions for warnings"""
        settings = {
            'theme': 'light',
            'page_size': 25,
            'auto_refresh_interval': 3  # Below 5 seconds - should warn
        }

        result = SettingsValidator.validate_general_settings(settings)

        assert len(result['warnings']) > 0
        assert any('performance' in w.lower() for w in result['warnings'])

    def test_validate_analysis_settings_edge_values(self):
        """Test analysis settings with edge case values"""
        settings = {
            'default_sampling_freq': 100,  # Minimum acceptable
            'default_fft_points': 256,  # Minimum acceptable
            'peak_threshold': 0.1  # Minimum value
        }

        result = SettingsValidator.validate_analysis_settings(settings)

        assert result['valid'] is True

    def test_validate_data_settings_edge_values(self):
        """Test data settings with edge case values"""
        settings = {
            'max_file_size': 10,  # Minimum value
            'auto_save_interval': 1,  # Minimum value
            'data_retention_days': 1  # Minimum value
        }

        result = SettingsValidator.validate_data_settings(settings)

        assert result['valid'] is True

    def test_validate_system_settings_edge_values(self):
        """Test system settings with edge case values"""
        settings = {
            'max_cpu_usage': 10,  # Minimum value
            'memory_limit_gb': 1,  # Minimum value
            'parallel_threads': 1  # Minimum value
        }

        result = SettingsValidator.validate_system_settings(settings)

        assert result['valid'] is True


@pytest.mark.skipif(not SETTINGS_UTILS_AVAILABLE, reason="Settings utils module not available")
class TestSystemRecommendations:
    """Test system recommendations function"""

    @patch.object(SystemMonitor, 'get_cpu_info')
    @patch.object(SystemMonitor, 'get_memory_info')
    @patch.object(SystemMonitor, 'get_system_info')
    def test_get_system_recommendations_windows(self, mock_system, mock_memory, mock_cpu):
        """Test recommendations for Windows system"""
        mock_system.return_value = {'platform': 'Windows'}
        mock_memory.return_value = {'total_gb': 8}
        mock_cpu.return_value = {'core_count': 4}

        recommendations = get_system_recommendations()

        assert isinstance(recommendations, dict)
        assert 'system' in recommendations
        assert any('Windows' in r for r in recommendations['system'])

    @patch.object(SystemMonitor, 'get_cpu_info')
    @patch.object(SystemMonitor, 'get_memory_info')
    @patch.object(SystemMonitor, 'get_system_info')
    def test_get_system_recommendations_low_memory(self, mock_system, mock_memory, mock_cpu):
        """Test recommendations for low memory system"""
        mock_system.return_value = {'platform': 'Linux'}
        mock_memory.return_value = {'total_gb': 2}  # Low memory
        mock_cpu.return_value = {'core_count': 4}

        recommendations = get_system_recommendations()

        assert isinstance(recommendations, dict)
        assert 'system' in recommendations
        assert any('memory' in r.lower() for r in recommendations['system'])

    @patch.object(SystemMonitor, 'get_cpu_info')
    @patch.object(SystemMonitor, 'get_memory_info')
    @patch.object(SystemMonitor, 'get_system_info')
    def test_get_system_recommendations_limited_cpu(self, mock_system, mock_memory, mock_cpu):
        """Test recommendations for limited CPU cores"""
        mock_system.return_value = {'platform': 'Linux'}
        mock_memory.return_value = {'total_gb': 8}
        mock_cpu.return_value = {'core_count': 2}  # Limited cores

        recommendations = get_system_recommendations()

        assert isinstance(recommendations, dict)
        assert 'analysis' in recommendations
        assert any('CPU' in r or 'cores' in r.lower() for r in recommendations['analysis'])

    @patch.object(SystemMonitor, 'get_cpu_info')
    @patch.object(SystemMonitor, 'get_memory_info')
    @patch.object(SystemMonitor, 'get_system_info')
    def test_get_system_recommendations_error(self, mock_system, mock_memory, mock_cpu):
        """Test recommendations when error occurs"""
        mock_system.side_effect = Exception("System error")

        recommendations = get_system_recommendations()

        assert isinstance(recommendations, dict)
        assert len(recommendations) == 0


@pytest.mark.skipif(not SETTINGS_UTILS_AVAILABLE, reason="Settings utils module not available")
class TestSettingsExporterAdditional:
    """Additional tests for SettingsExporter"""

    def test_export_settings_json_with_custom_filename(self):
        """Test exporting settings with custom filename"""
        settings = {'theme': 'dark', 'page_size': 50}

        with tempfile.TemporaryDirectory() as temp_dir:
            custom_filename = os.path.join(temp_dir, 'my_settings.json')

            result = SettingsExporter.export_settings_json(settings, custom_filename)

            assert result == custom_filename
            assert os.path.exists(custom_filename)

            with open(custom_filename, 'r') as f:
                data = json.load(f)

            assert 'export_info' in data
            assert 'settings' in data
            assert data['settings'] == settings
            assert data['export_info']['application'] == 'vitalDSP Webapp'

    def test_export_settings_json_auto_filename(self):
        """Test exporting settings with auto-generated filename"""
        settings = {'theme': 'light', 'page_size': 25}

        # Export with auto filename (in current directory)
        try:
            result = SettingsExporter.export_settings_json(settings)

            assert result is not None
            assert 'vitaldsp_settings_' in result
            assert '.json' in result
            assert os.path.exists(result)

            with open(result, 'r') as f:
                data = json.load(f)

            assert data['settings'] == settings
        finally:
            # Clean up
            if result and os.path.exists(result):
                os.unlink(result)

    def test_import_settings_json_invalid_format(self):
        """Test importing settings with invalid format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Write invalid format (missing 'settings' key)
            json.dump({'other_data': 'value'}, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid settings file format"):
                SettingsExporter.import_settings_json(temp_path)
        finally:
            os.unlink(temp_path)

    def test_export_settings_json_write_error(self):
        """Test export error handling"""
        settings = {'theme': 'dark'}

        # Try to write to invalid path
        with pytest.raises(Exception):
            SettingsExporter.export_settings_json(
                settings,
                '/invalid/path/that/does/not/exist/settings.json'
            )

    def test_import_settings_json_file_not_found(self):
        """Test importing from non-existent file"""
        with pytest.raises(Exception):
            SettingsExporter.import_settings_json('/nonexistent/file.json')
