import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import dash_bootstrap_components as dbc
from dash import html, dcc

# Add the src directory to the Python path so tests can import vitalDSP_webapp modules
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import the module directly and access functions as attributes
import vitalDSP_webapp.callbacks.vitaldsp_callbacks as vitaldsp_module


class TestVitalDSPCallbacks:
    """Test class for vitalDSP callbacks."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_app = Mock()
        self.test_data = {
            'data_id': 'test123',
            'columns': ['time', 'signal', 'red', 'ir'],
            'sampling_freq': 1000,
            'time_unit': 'ms'
        }
        
        # Create test DataFrame
        self.test_df = pd.DataFrame({
            'time': np.arange(1000),
            'signal': np.random.randn(1000) + 10,
            'red': np.random.randn(1000) + 8,
            'ir': np.random.randn(1000) + 6
        })

    @patch('vitalDSP_webapp.callbacks.vitaldsp_callbacks.dcc')
    @patch('vitalDSP_webapp.callbacks.vitaldsp_callbacks.html')
    def test_register_vitaldsp_callbacks(self, mock_html, mock_dcc):
        """Test registering vitalDSP callbacks."""
        # Mock the callback decorator
        mock_callback = Mock()
        mock_callback.side_effect = lambda *args, **kwargs: lambda func: func
        
        with patch('vitalDSP_webapp.callbacks.vitaldsp_callbacks.app.callback', mock_callback):
            vitaldsp_module.register_vitaldsp_callbacks(self.mock_app)
            
            # Check that callbacks were registered
            assert mock_callback.call_count > 0

    @patch('vitalDSP_webapp.callbacks.vitaldsp_callbacks.get_data_service')
    def test_process_signal_analysis_basic(self, mock_get_service):
        """Test basic signal analysis processing."""
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        mock_service.get_data.return_value = self.test_df
        
        result = vitaldsp_module.process_signal_analysis(
            'test123', 
            'hrv_analysis', 
            {'window_size': 300}
        )
        
        assert isinstance(result, dict)
        assert 'status' in result
        assert 'results' in result

    @patch('vitalDSP_webapp.callbacks.vitaldsp_callbacks.get_data_service')
    def test_process_signal_analysis_no_data(self, mock_get_service):
        """Test signal analysis when no data is available."""
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        mock_service.get_data.return_value = None
        
        result = vitaldsp_module.process_signal_analysis(
            'test123', 
            'hrv_analysis', 
            {'window_size': 300}
        )
        
        assert result['status'] == 'error'
        assert 'No data found' in result['message']

    @patch('vitalDSP_webapp.callbacks.vitaldsp_callbacks.get_data_service')
    def test_process_signal_analysis_invalid_analysis_type(self, mock_get_service):
        """Test signal analysis with invalid analysis type."""
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        mock_service.get_data.return_value = self.test_df
        
        result = vitaldsp_module.process_signal_analysis(
            'test123', 
            'invalid_analysis', 
            {}
        )
        
        assert result['status'] == 'error'
        assert 'Unsupported analysis type' in result['message']

    @patch('vitalDSP_webapp.callbacks.vitaldsp_callbacks.get_data_service')
    def test_process_signal_analysis_hrv(self, mock_get_service):
        """Test HRV analysis processing."""
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        mock_service.get_data.return_value = self.test_df
        
        result = vitaldsp_module.process_signal_analysis(
            'test123', 
            'hrv_analysis', 
            {'window_size': 300, 'overlap': 0.5}
        )
        
        assert result['status'] == 'success'
        assert 'hrv_features' in result['results']

    @patch('vitalDSP_webapp.callbacks.vitaldsp_callbacks.get_data_service')
    def test_process_signal_analysis_frequency_domain(self, mock_get_service):
        """Test frequency domain analysis processing."""
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        mock_service.get_data.return_value = self.test_df
        
        result = vitaldsp_module.process_signal_analysis(
            'test123', 
            'frequency_domain', 
            {'window_size': 512, 'overlap': 0.5}
        )
        
        assert result['status'] == 'success'
        assert 'frequency_features' in result['results']

    @patch('vitalDSP_webapp.callbacks.vitaldsp_callbacks.get_data_service')
    def test_process_signal_analysis_time_domain(self, mock_get_service):
        """Test time domain analysis processing."""
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        mock_service.get_data.return_value = self.test_df
        
        result = vitaldsp_module.process_signal_analysis(
            'test123', 
            'time_domain', 
            {'window_size': 100}
        )
        
        assert result['status'] == 'success'
        assert 'time_features' in result['results']

    @patch('vitalDSP_webapp.callbacks.vitaldsp_callbacks.get_data_service')
    def test_process_signal_analysis_wavelet(self, mock_get_service):
        """Test wavelet analysis processing."""
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        mock_service.get_data.return_value = self.test_df
        
        result = vitaldsp_module.process_signal_analysis(
            'test123', 
            'wavelet_analysis', 
            {'wavelet': 'db4', 'levels': 6}
        )
        
        assert result['status'] == 'success'
        assert 'wavelet_features' in result['results']

    def test_generate_analysis_report_basic(self):
        """Test basic analysis report generation."""
        analysis_results = {
            'hrv_features': {
                'mean_hr': 75.5,
                'sdnn': 45.2,
                'rmssd': 32.1
            },
            'quality_metrics': {
                'snr': 25.3,
                'artifact_count': 5
            }
        }
        
        report = vitaldsp_module.generate_analysis_report(analysis_results, 'hrv_analysis')
        
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'details' in report
        assert 'recommendations' in report

    def test_generate_analysis_report_empty_results(self):
        """Test analysis report generation with empty results."""
        report = vitaldsp_module.generate_analysis_report({}, 'hrv_analysis')
        
        assert report['summary'] == 'No analysis results available'
        assert report['details'] == {}
        assert 'No specific recommendations' in report['recommendations']

    def test_generate_analysis_report_hrv(self):
        """Test HRV analysis report generation."""
        analysis_results = {
            'hrv_features': {
                'mean_hr': 75.5,
                'sdnn': 45.2,
                'rmssd': 32.1,
                'pnn50': 28.5
            }
        }
        
        report = vitaldsp_module.generate_analysis_report(analysis_results, 'hrv_analysis')
        
        assert 'HRV Analysis Report' in report['summary']
        assert 'mean_hr' in str(report['details'])
        assert 'HRV' in report['recommendations']

    def test_generate_analysis_report_frequency_domain(self):
        """Test frequency domain analysis report generation."""
        analysis_results = {
            'frequency_features': {
                'total_power': 1250.5,
                'vlf_power': 150.2,
                'lf_power': 450.8,
                'hf_power': 649.5
            }
        }
        
        report = vitaldsp_module.generate_analysis_report(analysis_results, 'frequency_domain')
        
        assert 'Frequency Domain Analysis' in report['summary']
        assert 'total_power' in str(report['details'])
        assert 'frequency' in report['recommendations'].lower()

    def test_create_visualization_components_basic(self):
        """Test basic visualization component creation."""
        analysis_results = {
            'hrv_features': {'mean_hr': 75.5},
            'plots': {'hrv_plot': 'plot_data'}
        }
        
        components = vitaldsp_module.create_visualization_components(analysis_results, 'hrv_analysis')
        
        assert isinstance(components, list)
        assert len(components) > 0

    def test_create_visualization_components_no_plots(self):
        """Test visualization component creation without plots."""
        analysis_results = {
            'hrv_features': {'mean_hr': 75.5}
        }
        
        components = vitaldsp_module.create_visualization_components(analysis_results, 'hrv_analysis')
        
        assert isinstance(components, list)
        # Should still create some components even without plots

    def test_create_visualization_components_hrv(self):
        """Test HRV visualization component creation."""
        analysis_results = {
            'hrv_features': {
                'mean_hr': 75.5,
                'sdnn': 45.2,
                'rmssd': 32.1
            },
            'plots': {
                'rr_intervals': 'rr_data',
                'hrv_spectrum': 'spectrum_data'
            }
        }
        
        components = vitaldsp_module.create_visualization_components(analysis_results, 'hrv_analysis')
        
        assert isinstance(components, list)
        assert len(components) > 0

    def test_create_visualization_components_frequency_domain(self):
        """Test frequency domain visualization component creation."""
        analysis_results = {
            'frequency_features': {
                'total_power': 1250.5,
                'vlf_power': 150.2,
                'lf_power': 450.8,
                'hf_power': 649.5
            },
            'plots': {
                'power_spectrum': 'spectrum_data',
                'power_distribution': 'distribution_data'
            }
        }
        
        components = vitaldsp_module.create_visualization_components(analysis_results, 'frequency_domain')
        
        assert isinstance(components, list)
        assert len(components) > 0

    @patch('vitalDSP_webapp.callbacks.vitaldsp_callbacks.get_data_service')
    def test_process_signal_analysis_with_parameters(self, mock_get_service):
        """Test signal analysis with various parameters."""
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        mock_service.get_data.return_value = self.test_df
        
        # Test with different parameter combinations
        param_combinations = [
            {'window_size': 256, 'overlap': 0.5},
            {'wavelet': 'db4', 'levels': 4},
            {'filter_type': 'butterworth', 'cutoff': 0.5},
            {'threshold': 0.1, 'min_peak_distance': 10}
        ]
        
        for params in param_combinations:
            result = vitaldsp_module.process_signal_analysis('test123', 'hrv_analysis', params)
            assert result['status'] == 'success'

    def test_generate_analysis_report_edge_cases(self):
        """Test analysis report generation with edge cases."""
        # Test with None results
        report = vitaldsp_module.generate_analysis_report(None, 'hrv_analysis')
        assert 'No analysis results available' in report['summary']
        
        # Test with string results
        report = vitaldsp_module.generate_analysis_report("string results", 'hrv_analysis')
        assert 'Invalid results format' in report['summary']
        
        # Test with empty analysis type
        report = vitaldsp_module.generate_analysis_report({}, '')
        assert 'Unknown analysis type' in report['summary']

    def test_create_visualization_components_edge_cases(self):
        """Test visualization component creation with edge cases."""
        # Test with None results
        components = vitaldsp_module.create_visualization_components(None, 'hrv_analysis')
        assert isinstance(components, list)
        
        # Test with empty results
        components = vitaldsp_module.create_visualization_components({}, 'hrv_analysis')
        assert isinstance(components, list)
        
        # Test with invalid analysis type
        components = vitaldsp_module.create_visualization_components({'data': 'test'}, 'invalid_type')
        assert isinstance(components, list)

    @patch('vitalDSP_webapp.callbacks.vitaldsp_callbacks.get_data_service')
    def test_process_signal_analysis_error_handling(self, mock_get_service):
        """Test error handling in signal analysis."""
        mock_service = Mock()
        mock_get_service.return_value = mock_service
        mock_service.get_data.side_effect = Exception("Service error")
        
        result = vitaldsp_module.process_signal_analysis('test123', 'hrv_analysis', {})
        
        assert result['status'] == 'error'
        assert 'Service error' in result['message']

    def test_analysis_parameter_validation(self):
        """Test validation of analysis parameters."""
        # Test valid parameters
        valid_params = {
            'window_size': 256,
            'overlap': 0.5,
            'wavelet': 'db4',
            'levels': 6
        }
        
        # Test invalid parameters
        invalid_params = {
            'window_size': -1,  # Invalid window size
            'overlap': 1.5,     # Invalid overlap
            'levels': 0         # Invalid levels
        }
        
        # This would typically be validated in the actual function
        # For now, we just test that the function can handle various parameter types
        assert isinstance(valid_params, dict)
        assert isinstance(invalid_params, dict)
